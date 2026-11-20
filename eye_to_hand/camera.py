"""Camera intrinsics, checkerboard detection and planar pose estimation.

This module covers the "camera side" of the pipeline:

* describe a pinhole camera (intrinsics + distortion),
* detect a checkerboard (``detectCheckerboardPoints``),
* estimate ``T_board_to_camera`` with a planar PnP solver
  (``estimateExtrinsics``),
* optionally calibrate the intrinsics from a set of checkerboard images.

Corner coordinates are always ``(N, 2)`` pixel coordinates in the image that
was passed to the detector.  When the image was undistorted first, pass
:meth:`CameraIntrinsics.without_distortion` to the PnP solver (see
:func:`estimate_board_pose`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .transforms import make_transform, rotation_to_rotvec

__all__ = [
    "CameraIntrinsics",
    "Checkerboard",
    "BoardPose",
    "IntrinsicsCalibration",
    "detect_checkerboard",
    "estimate_board_pose",
    "calibrate_intrinsics",
    "undistort_image",
    "save_intrinsics",
    "load_intrinsics",
]


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    """Pinhole camera parameters in OpenCV convention."""

    K: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.distortion = np.asarray(self.distortion, dtype=float).ravel()
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    def without_distortion(self) -> "CameraIntrinsics":
        """Copy with zero lens distortion (for already-undistorted images)."""
        return CameraIntrinsics(self.K.copy(), np.zeros_like(self.distortion), self.image_size)

    def project(self, points_camera: np.ndarray, apply_distortion: bool = True) -> np.ndarray:
        """Project ``(N, 3)`` points expressed in the camera frame to pixels."""
        points = np.asarray(points_camera, dtype=float).reshape(-1, 1, 3)
        dist = self.distortion if apply_distortion else np.zeros_like(self.distortion)
        if dist.size == 0:
            dist = np.zeros(5)
        uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.K, dist)
        return uv.reshape(-1, 2)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return undistort_image(image, self)

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "distortion": self.distortion.tolist(),
            "image_size": [self.width, self.height],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        return cls(
            K=np.asarray(data["K"], dtype=float),
            distortion=np.asarray(data.get("distortion", []), dtype=float),
            image_size=tuple(data["image_size"]),
        )


def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Undistort an image; a no-op when all distortion coefficients are zero."""
    if not np.any(intrinsics.distortion):
        return image
    return cv2.undistort(image, intrinsics.K, intrinsics.distortion)


# ---------------------------------------------------------------------------
# Checkerboard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Checkerboard:
    """Planar checkerboard target.

    ``columns`` and ``rows`` count **inner** corners (the OpenCV convention),
    not squares.  A classic 9x6 board therefore has 10x7 squares.
    """

    columns: int = 9
    rows: int = 6
    square_size: float = 0.02  # metres

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise ValueError("checkerboard needs at least 2x2 inner corners")
        if self.square_size <= 0:
            raise ValueError("square_size must be positive")

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.columns, self.rows

    @property
    def n_corners(self) -> int:
        return self.columns * self.rows

    def object_points(self) -> np.ndarray:
        """``(N, 3)`` board-frame coordinates of the inner corners (z = 0)."""
        cols, rows = self.columns, self.rows
        points = np.zeros((cols * rows, 3), dtype=float)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return points * self.square_size

    @classmethod
    def from_string(cls, text: str, square_size: float = 0.02) -> "Checkerboard":
        try:
            cols, rows = (int(v) for v in text.lower().split("x"))
        except ValueError as exc:
            raise ValueError("board must look like '9x6'") from exc
        return cls(cols, rows, square_size)

    def to_dict(self) -> dict:
        return {"columns": self.columns, "rows": self.rows, "square_size": self.square_size}

    @classmethod
    def from_dict(cls, data: dict) -> "Checkerboard":
        return cls(
            columns=int(data.get("columns", data.get("board_pattern_size", [9, 6])[0])),
            rows=int(data.get("rows", data.get("board_pattern_size", [9, 6])[1])),
            square_size=float(data.get("square_size", data.get("board_square_size", 0.02))),
        )


# ---------------------------------------------------------------------------
# Detection and PnP
# ---------------------------------------------------------------------------
def detect_checkerboard(
    image: np.ndarray,
    board: Checkerboard,
    refine: bool = True,
) -> np.ndarray | None:
    """Detect all inner corners of ``board`` in ``image``.

    Returns ``None`` when the board is not fully visible.  This mirrors
    MATLAB's ``detectCheckerboardPoints(..., PartialDetections=false)``:
    partial boards are rejected instead of being silently used.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, board.pattern_size, flags)

    if not found and hasattr(cv2, "findChessboardCornersSB"):
        # The SB detector is slower but handles blur / low contrast better.
        found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size)

    if not found or corners is None:
        return None

    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(corners, dtype=float).reshape(-1, 2)


@dataclass
class BoardPose:
    T_board_to_camera: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    reprojection_rmse_px: float
    rvec: np.ndarray
    tvec: np.ndarray


def estimate_board_pose(
    image_points: np.ndarray,
    board: Checkerboard,
    intrinsics: CameraIntrinsics,
) -> BoardPose:
    """Estimate ``T_board_to_camera`` from planar correspondences.

    ``image_points`` must correspond to an image without lens distortion
    (call :meth:`CameraIntrinsics.without_distortion` first), otherwise the
    PnP residual is biased by the distortion model.
    """
    object_points = board.object_points()
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(image_points) != len(object_points):
        raise ValueError(f"expected {len(object_points)} corners, got {len(image_points)}")

    dist = np.zeros_like(intrinsics.distortion) if intrinsics.distortion.size else np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.K,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        raise RuntimeError("solvePnP failed")

    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, intrinsics.K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = make_transform(R, np.asarray(tvec).ravel())

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics.K, dist)
    residual = projected.reshape(-1, 2) - image_points
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return BoardPose(
        T_board_to_camera=T,
        image_points=image_points,
        object_points=object_points,
        reprojection_rmse_px=rmse,
        rvec=np.asarray(rvec).ravel(),
        tvec=np.asarray(tvec).ravel(),
    )


# ---------------------------------------------------------------------------
# Intrinsics calibration
# ---------------------------------------------------------------------------
@dataclass
class IntrinsicsCalibration:
    intrinsics: CameraIntrinsics
    rms_reprojection_px: float
    per_view_rmse_px: np.ndarray
    used_images: list[Path]
    board: Checkerboard


def calibrate_intrinsics(
    image_paths: Iterable[str | Path],
    board: Checkerboard,
    image_size: tuple[int, int] | None = None,
    max_views: int | None = None,
) -> IntrinsicsCalibration:
    """Estimate intrinsics + distortion with ``cv2.calibrateCamera``.

    The checkerboard images are detected on the **raw** images; OpenCV fits
    the distortion model at the same time.  At least four valid views are
    required.
    """
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used: list[Path] = []

    for path in map(Path, image_paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        corners = detect_checkerboard(image, board)
        if corners is None:
            continue
        object_points.append(board.object_points())
        image_points.append(corners)
        used.append(path)
        if max_views is not None and len(used) >= max_views:
            break

    if len(used) < 4:
        raise RuntimeError(f"need at least 4 valid checkerboard views, got {len(used)}")
    if image_size is None:
        raise RuntimeError("could not determine image size")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None, criteria=criteria)

    # calibrateCamera does not expose per-view rvec/tvec in this call form,
    # so recompute each view's reprojection error with solvePnP using the
    # fitted intrinsics.
    per_view = []
    for obj, img in zip(object_points, image_points):
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            per_view.append(float("nan"))
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = projected.reshape(-1, 2) - img.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))

    return IntrinsicsCalibration(
        intrinsics=CameraIntrinsics(K=K, distortion=np.asarray(dist).ravel(), image_size=image_size),
        rms_reprojection_px=float(rms),
        per_view_rmse_px=np.asarray(per_view, dtype=float),
        used_images=used,
        board=board,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_intrinsics(
    path: str | Path,
    intrinsics: CameraIntrinsics,
    board: Checkerboard | None = None,
) -> None:
    """Save intrinsics to ``.npz`` (pipeline format) or JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        payload = {
            "K": intrinsics.K,
            "distortion": intrinsics.distortion,
            "image_size": np.asarray(intrinsics.image_size, dtype=int),
        }
        if board is not None:
            payload["board_pattern_size"] = np.asarray(board.pattern_size, dtype=int)
            payload["board_square_size"] = float(board.square_size)
        np.savez(path, **payload)
    else:
        import json

        data = intrinsics.to_dict()
        if board is not None:
            data["board"] = board.to_dict()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_intrinsics(path: str | Path) -> tuple[CameraIntrinsics, Checkerboard | None]:
    """Load intrinsics from ``.npz`` or JSON; also returns board metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npz":
        data = np.load(path)
        intrinsics = CameraIntrinsics(
            K=data["K"],
            distortion=data["distortion"],
            image_size=tuple(int(v) for v in data["image_size"]),
        )
        board = None
        if "board_pattern_size" in data:
            board = Checkerboard(
                columns=int(data["board_pattern_size"][0]),
                rows=int(data["board_pattern_size"][1]),
                square_size=float(data["board_square_size"]),
            )
        return intrinsics, board

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics.from_dict(data)
    board = Checkerboard.from_dict(data["board"]) if "board" in data else None
    return intrinsics, board

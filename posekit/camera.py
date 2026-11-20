"""Camera intrinsics and checkerboard extrinsics.

Python equivalents of:
    undistortImage
    detectCheckerboardPoints(img, PartialDetections=false)
    patternWorldPoints("checkerboard", boardsz, squareSize)
    estimateExtrinsics(imgpts, worldpts, intrinsics)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .se3 import make_transform

# OpenCV's classic 9x6 chessboard (same board as the MathWorks example).
DEFAULT_BOARD = (9, 6)  # inner corners: columns, rows
DEFAULT_SQUARE_SIZE = 0.02  # meters (25 mm board -> 20 mm inner square)


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    @classmethod
    def from_calibration_file(cls, path: str | Path) -> "CameraIntrinsics":
        data = np.load(path)
        return cls(K=data["K"], distortion=data["distortion"], image_size=tuple(int(v) for v in data["image_size"]))

    @classmethod
    def from_opencv(cls, camera_matrix, dist_coeffs, image_size) -> "CameraIntrinsics":
        return cls(K=np.asarray(camera_matrix, float), distortion=np.asarray(dist_coeffs, float).ravel(), image_size=image_size)

    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @property
    def cx(self):
        return float(self.K[0, 2])

    @property
    def cy(self):
        return float(self.K[1, 2])

    def project(self, T_cam_to_world_points: np.ndarray) -> np.ndarray:
        """Project (N, 3) points given in the camera frame to pixels."""
        pts = np.asarray(T_cam_to_world_points, float).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.distortion)
        return uv.reshape(-1, 2)


@dataclass
class Checkerboard:
    pattern_size: tuple[int, int] = DEFAULT_BOARD  # (columns, rows) of inner corners
    square_size: float = DEFAULT_SQUARE_SIZE

    def world_points(self) -> np.ndarray:
        """Object points (N, 3) in the board frame, z = 0."""
        cols, rows = self.pattern_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return objp * self.square_size


@dataclass
class ExtrinsicsResult:
    image_points: np.ndarray
    world_points: np.ndarray
    T_board_to_camera: np.ndarray  # 4x4, camera <- board
    reprojection_rmse: float


def detect_checkerboard(image: np.ndarray, board: Checkerboard) -> np.ndarray | None:
    """Detect inner corners, discard partial detections (PartialDetections=false)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, board.pattern_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def estimate_extrinsics(
    image_points: np.ndarray,
    world_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> ExtrinsicsResult:
    """Estimate the camera extrinsics (board -> camera) with IPPE.

    IPPE is the recommended solver for a planar checkerboard.  ``solvePnP``
    returns OpenCV's preferred branch; use ``solvePnPGeneric`` if you need
    to inspect both solutions.  Subsequent hand-eye refinement removes the
    small pose bias caused by corner noise.
    """
    ok, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        intrinsics.K,
        intrinsics.distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    # Refine with Levenberg-Marquardt on all points.
    rvec, tvec = cv2.solvePnPRefineLM(
        world_points, image_points, intrinsics.K, intrinsics.distortion, rvec, tvec
    )

    R, _ = cv2.Rodrigues(rvec)
    T = make_transform(R, tvec.ravel())

    projected, _ = cv2.projectPoints(world_points, rvec, tvec, intrinsics.K, intrinsics.distortion)
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return ExtrinsicsResult(
        image_points=image_points.reshape(-1, 2),
        world_points=world_points,
        T_board_to_camera=T,
        reprojection_rmse=rmse,
    )


def compute_camera_extrinsics(
    image_paths,
    intrinsics: CameraIntrinsics,
    board: Checkerboard = Checkerboard(),
    verbose: bool = True,
) -> list[ExtrinsicsResult | None]:
    """Loop over the calibration images (MATLAB: for r = 1:numimgs ...).

    The image is undistorted first (like ``undistortImage``), so the PnP solve
    must use a *zero-distortion* copy of the intrinsics: the detected corners
    are already in ideal pixel coordinates.
    """
    undistorted_intrinsics = CameraIntrinsics(
        K=intrinsics.K,
        distortion=np.zeros_like(intrinsics.distortion),
        image_size=intrinsics.image_size,
    )
    results: list[ExtrinsicsResult | None] = []
    for path in image_paths:
        path = Path(path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = undistort_image(image, intrinsics)
        corners = detect_checkerboard(image, board)
        if corners is None:
            if verbose:
                print(f"[warn] checkerboard not found (or partially detected): {path.name}")
            results.append(None)
            continue
        result = estimate_extrinsics(corners, board.world_points(), undistorted_intrinsics)
        if verbose:
            print(f"[ok]   {path.name}: reprojection RMSE = {result.reprojection_rmse:.3f} px")
        results.append(result)
    return results


def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """MATLAB: undistortImage."""
    if not np.any(intrinsics.distortion):
        return image
    return cv2.undistort(image, intrinsics.K, intrinsics.distortion)

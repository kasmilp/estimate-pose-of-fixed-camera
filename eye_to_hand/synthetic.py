"""Synthetic eye-to-hand dataset generator with a known ground truth.

The scene mirrors the real setup: a Kinova Gen3 is mounted on a table, a
checkerboard is bolted to its end effector, and a camera on a fixed beam
looks down at the workspace.  The 15 joint configurations are the ones from
the MathWorks example, so the generated dataset exercises the same motion.

The renderer is intentionally simple (no meshes, no lighting): its purpose is
to produce images that a real checkerboard detector can process, so the full
pipeline can be verified without downloading the MathWorks dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraIntrinsics, Checkerboard
from .dataset import save_dataset_metadata
from .kinematics import RobotModel
from .transforms import (
    inverse,
    make_transform,
    rpy_to_rotation,
    transform_points,
)

__all__ = [
    "JOINT_ANGLES_DEG",
    "SyntheticScene",
    "generate_synthetic_dataset",
]

# Joint angles (degrees) from jointAngles.mat of the MathWorks example.
JOINT_ANGLES_DEG = np.array(
    [
        [346.8241, 352.5997, 191.5695, 212.1722, 355.5775, 38.5151, 184.8700],
        [346.8241, 352.5996, 191.5695, 212.1724, 355.5776, 20.6987, 184.8699],
        [346.8241, 352.5996, 191.5695, 212.1723, 355.5775, 55.1762, 184.8699],
        [346.8241, 352.5996, 191.5695, 212.1723, 355.5775, 55.1761, 159.7137],
        [344.9588, 352.5997, 191.5695, 212.1723, 355.5775, 47.9125, 202.4223],
        [344.9591, 350.7478, 191.5696, 212.1723, 334.9004, 43.3278, 191.6003],
        [344.9592, 350.7479, 191.5696, 212.1722, 22.5804, 43.3278, 191.6003],
        [344.9592, 350.7477, 191.5696, 212.1723, 14.6105, 17.6080, 191.6004],
        [344.9592, 350.7478, 191.5696, 212.1723, 65.1178, 27.7940, 121.8693],
        [9.4972, 350.7477, 191.5696, 212.1723, 65.1178, 27.7940, 121.8693],
        [2.4139, 356.8379, 179.3088, 225.4389, 79.5231, 35.3008, 125.8531],
        [15.2262, 356.8380, 179.3092, 225.4386, 48.9265, 55.5291, 125.8531],
        [15.2261, 4.5792, 179.3092, 225.4386, 48.9264, 55.5291, 175.2737],
        [31.0168, 4.5793, 179.3092, 225.4386, 48.9263, 55.5291, 174.4471],
        [34.9532, 358.5155, 179.3093, 225.4386, 48.9264, 55.5294, 145.9294],
    ]
)


@dataclass
class SyntheticScene:
    intrinsics: CameraIntrinsics
    T_camera_to_base: np.ndarray
    T_board_to_end_effector: np.ndarray
    board: Checkerboard

    @classmethod
    def default(cls, image_size: tuple[int, int] = (1280, 720), seed: int = 0) -> "SyntheticScene":
        rng = np.random.default_rng(seed)
        width, height = image_size
        intrinsics = CameraIntrinsics(
            K=np.array(
                [
                    [900.0, 0.0, width / 2 + rng.uniform(-20, 20)],
                    [0.0, 905.0, height / 2 + rng.uniform(-15, 15)],
                    [0.0, 0.0, 1.0],
                ]
            ),
            distortion=np.array([-0.08, 0.06, 0.0005, -0.0004, 0.0]),
            image_size=image_size,
        )
        # Camera on a beam above the workspace, optical +Z pointing down.
        T_camera_to_base = make_transform(
            rpy_to_rotation(np.deg2rad([180.0, 0.0, 350.0])),
            [0.32, 0.0, 1.44],
        )
        # Board bolted to the tool flange through a small bracket.
        T_board_to_end_effector = make_transform(
            rpy_to_rotation(np.deg2rad([0.0, 90.0, 0.0])),
            [0.0, 0.0, 0.04],
        )
        return cls(intrinsics, T_camera_to_base, T_board_to_end_effector, Checkerboard())


def generate_synthetic_dataset(
    out_dir: str | Path,
    scene: SyntheticScene | None = None,
    joint_angles_deg: np.ndarray = JOINT_ANGLES_DEG,
    image_noise: float = 2.0,
    blur_sigma: float = 0.6,
    seed: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """Render images and write a complete calibration dataset.

    Parameters
    ----------
    out_dir : path
        Destination directory.
    scene : SyntheticScene, optional
        Camera / board geometry.  The default scene is tuned so the board is
        visible in all 15 poses.
    joint_angles_deg : (N, 7) array
        Robot configurations, in degrees.
    image_noise : float
        Standard deviation of additive Gaussian pixel noise (gray levels).
    blur_sigma : float
        Gaussian blur sigma applied after rendering (pixels).
    seed : int
        Seed for reproducible noise.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{images_dir} already contains images; pass overwrite=True")
    images_dir.mkdir(parents=True, exist_ok=True)

    scene = scene or SyntheticScene.default()
    model = RobotModel()
    rng = np.random.default_rng(seed)
    angles = np.atleast_2d(np.asarray(joint_angles_deg, dtype=float))
    if angles.shape[1] != model.n_dof:
        raise ValueError(f"expected {model.n_dof} joint angles per row, got {angles.shape[1]}")

    world_points = scene.board.object_points()
    T_base_to_camera = inverse(scene.T_camera_to_base)
    poses: list[dict] = []

    for index, row in enumerate(angles, start=1):
        T_ee_to_base = model.fk_deg(row)
        T_board_to_base = T_ee_to_base @ scene.T_board_to_end_effector
        T_board_to_camera = T_base_to_camera @ T_board_to_base

        points_camera = transform_points(T_board_to_camera, world_points)
        uv = scene.intrinsics.project(points_camera)
        width, height = scene.intrinsics.image_size
        visible = bool(
            (points_camera[:, 2] > 0.05).all()
            and (uv[:, 0] > 8).all()
            and (uv[:, 0] < width - 8).all()
            and (uv[:, 1] > 8).all()
            and (uv[:, 1] < height - 8).all()
        )

        image = _render_scene(
            scene=scene,
            model=model,
            joint_angles_deg=row,
            T_ee_to_base=T_ee_to_base,
            T_board_to_camera=T_board_to_camera,
            visible=visible,
        )
        image = _add_sensor_noise(image, rng, image_noise, blur_sigma)
        cv2.imwrite(str(images_dir / f"{index}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        poses.append(
            {
                "index": index,
                "joint_angles_deg": row,
                "T_ee_to_base": T_ee_to_base,
                "T_board_to_base": T_board_to_base,
                "T_board_to_camera": T_board_to_camera,
                "visible": visible,
            }
        )
        if verbose:
            flag = "visible" if visible else "NOT VISIBLE"
            print(f"[generate] view {index:2d}/{len(angles)}  board {flag}")

    ground_truth = {
        "T_camera_to_base": scene.T_camera_to_base,
        "T_board_to_end_effector": scene.T_board_to_end_effector,
        "T_board_to_camera": np.stack([p["T_board_to_camera"] for p in poses]),
        "T_ee_to_base": np.stack([p["T_ee_to_base"] for p in poses]),
        "T_board_to_base": np.stack([p["T_board_to_base"] for p in poses]),
        "world_points": world_points,
        "visible": np.array([p["visible"] for p in poses]),
        "joint_angles_deg": angles,
    }
    save_dataset_metadata(
        out_dir,
        scene.intrinsics,
        scene.board,
        angles,
        ground_truth=ground_truth,
        extra={
            "generator": "eye_to_hand.synthetic",
            "seed": seed,
            "image_noise": image_noise,
            "blur_sigma": blur_sigma,
            "description": "Synthetic Kinova Gen3 eye-to-hand dataset with known ground truth",
        },
    )

    visible_count = int(np.sum([p["visible"] for p in poses]))
    if verbose:
        print(f"[generate] wrote {len(poses)} images to {images_dir} ({visible_count} fully visible)")

    return {
        "out_dir": out_dir,
        "scene": scene,
        "poses": poses,
        "visible_count": visible_count,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render_scene(
    scene: SyntheticScene,
    model: RobotModel,
    joint_angles_deg: np.ndarray,
    T_ee_to_base: np.ndarray,
    T_board_to_camera: np.ndarray,
    visible: bool,
) -> np.ndarray:
    width, height = scene.intrinsics.image_size
    image = _background(width, height)

    T_base_to_camera = inverse(scene.T_camera_to_base)

    # Ground grid.
    for value in np.arange(-0.7, 0.71, 0.1):
        _draw_line(image, scene.intrinsics, T_base_to_camera, [[value, -0.7, 0.0], [value, 0.7, 0.0]], (58, 58, 58), 1)
        _draw_line(image, scene.intrinsics, T_base_to_camera, [[-0.7, value, 0.0], [0.7, value, 0.0]], (58, 58, 58), 1)

    # Robot base plate.
    _draw_box(image, scene.intrinsics, T_base_to_camera, (-0.16, 0.16, -0.16, 0.16, 0.0, 0.035), (95, 95, 100))

    # Robot skeleton: links and joints.
    frames = model.link_frames(np.deg2rad(joint_angles_deg))
    ordered = list(frames.values())
    for a, b in zip(ordered[:-1], ordered[1:]):
        _draw_line(image, scene.intrinsics, T_base_to_camera, [a[:3, 3], b[:3, 3]], (185, 185, 190), 12)
    for frame in ordered[1:]:
        _draw_sphere(image, scene.intrinsics, T_base_to_camera, frame[:3, 3], 0.028, (215, 215, 220))

    # End-effector / board mount.
    if visible:
        _draw_checkerboard(image, scene.intrinsics, T_board_to_camera, scene.board)
        _draw_line(
            image,
            scene.intrinsics,
            T_base_to_camera,
            [T_ee_to_base[:3, 3], (T_ee_to_base @ scene.T_board_to_end_effector)[:3, 3]],
            (120, 120, 125),
            8,
        )

    return image


def _background(width: int, height: int) -> np.ndarray:
    column = np.linspace(28, 52, height, dtype=np.float32)[:, None]
    image = np.repeat(column, width, axis=1)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _draw_checkerboard(image: np.ndarray, intrinsics: CameraIntrinsics, T_board_to_camera: np.ndarray, board: Checkerboard) -> None:
    cols, rows = board.pattern_size
    square = board.square_size

    # The board's printed area extends one square beyond the inner corners.
    grid_world = np.zeros((rows + 2, cols + 2, 3), dtype=float)
    for r in range(rows + 2):
        for c in range(cols + 2):
            grid_world[r, c, :2] = ((c - 1) * square, (r - 1) * square)

    points_camera = transform_points(T_board_to_camera, grid_world.reshape(-1, 3))
    if (points_camera[:, 2] <= 0.05).any():
        return
    grid = intrinsics.project(points_camera).reshape(rows + 2, cols + 2, 2)

    # Thin dark backing plate.
    plate = np.array(
        [
            grid[0, 0],
            grid[0, -1],
            grid[-1, -1],
            grid[-1, 0],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, plate, (35, 35, 35), lineType=cv2.LINE_AA)

    for r in range(rows + 1):
        for c in range(cols + 1):
            quad = np.array(
                [grid[r, c], grid[r, c + 1], grid[r + 1, c + 1], grid[r + 1, c]],
                dtype=np.int32,
            )
            # Parity matches Checkerboard.object_points() / OpenCV's corner
            # ordering (verified end-to-end by the test-suite).
            color = (18, 18, 18) if (r + c) % 2 == 0 else (240, 240, 240)
            cv2.fillConvexPoly(image, quad, color, lineType=cv2.LINE_AA)


def _draw_line(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_to_camera: np.ndarray,
    points_base,
    color,
    thickness: int = 2,
) -> None:
    points_camera = transform_points(T_base_to_camera, np.asarray(points_base, dtype=float).reshape(-1, 3))
    if (points_camera[:, 2] < 0.05).any():
        return
    uv = intrinsics.project(points_camera).astype(int)
    cv2.line(image, tuple(uv[0]), tuple(uv[1]), color, thickness, cv2.LINE_AA)


def _draw_sphere(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_to_camera: np.ndarray,
    center_base,
    radius_m: float,
    color,
) -> None:
    center_camera = transform_points(T_base_to_camera, np.asarray(center_base, dtype=float).reshape(1, 3))[0]
    if center_camera[2] < 0.05:
        return
    uv = intrinsics.project(center_camera.reshape(1, 3))[0].astype(int)
    radius_px = max(2, int(intrinsics.fx * radius_m / center_camera[2]))
    cv2.circle(image, tuple(uv), radius_px, color, -1, cv2.LINE_AA)


def _draw_box(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_base_to_camera: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    color,
) -> None:
    x0, x1, y0, y1, z0, z1 = bounds
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        _draw_line(image, intrinsics, T_base_to_camera, [corners[a], corners[b]], color, 2)


def _add_sensor_noise(image: np.ndarray, rng: np.random.Generator, noise: float, blur_sigma: float) -> np.ndarray:
    result = image.astype(np.float32)
    if blur_sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), blur_sigma)
    if noise > 0:
        result += rng.normal(0.0, noise, result.shape)
    return np.clip(result, 0, 255).astype(np.uint8)

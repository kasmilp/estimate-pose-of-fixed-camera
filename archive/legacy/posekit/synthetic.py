"""Synthetic eye-to-hand dataset generator.

Purpose: validate the whole pipeline (intrinsics -> extrinsics -> hand-eye)
with a *known* ground truth when the MathWorks ``calibrationData.zip`` is not
available. Every calibration image is rendered by moving the KINOVA Gen3
to a joint configuration taken from ``jointAngles.mat`` and drawing the
checkerboard that is physically mounted on the end effector.

Physical model (same as the real setup):
    T_board_to_camera = T_camera_to_base^-1 @ T_end_effector_to_base @ T_board_to_end_effector
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraIntrinsics, Checkerboard
from .robot import RobotModel
from .se3 import euler_zyx_deg, invert, make_transform, rpy_to_matrix, transform_points

# Joint angles (degrees) copied from jointAngles.mat of the MathWorks example.
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
    def default(cls, image_size=(1280, 720), seed: int = 0) -> "SyntheticScene":
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
        # Camera mounted on a beam above the workspace, looking down
        # (its optical +Z axis is base -Z).
        T_camera_to_base = make_transform(
            rpy_to_matrix(np.deg2rad([180.0, 0.0, 350.0])),
            [0.32, 0.0, 1.44],
        )
        # Board bolted on the tool flange through a small bracket.
        T_board_to_end_effector = make_transform(
            rpy_to_matrix(np.deg2rad([0.0, 90.0, 0.0])),
            [0.0, 0.0, 0.04],
        )
        return cls(intrinsics, T_camera_to_base, T_board_to_end_effector, Checkerboard())


def generate(
    out_dir: str | Path,
    scene: SyntheticScene | None = None,
    joint_angles_deg: np.ndarray = JOINT_ANGLES_DEG,
    noise_px: float = 0.3,
    seed: int = 1,
) -> dict:
    """Render calibration images + write intrinsics.npz / joint_angles.txt."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scene = scene or SyntheticScene.default()
    model = RobotModel()
    rng = np.random.default_rng(seed)

    world_pts = scene.board.world_points()

    poses = []
    for idx, row in enumerate(joint_angles_deg, start=1):
        T_ee_to_base = model.fk(np.deg2rad(row))
        T_board_now = T_ee_to_base @ scene.T_board_to_end_effector

        # camera_T_board = camera_T_base @ base_T_board
        T_board_to_camera = invert(scene.T_camera_to_base) @ T_board_now
        pts_cam = transform_points(T_board_to_camera, world_pts)
        uv = scene.intrinsics.project(pts_cam)
        uv += rng.normal(0.0, noise_px, uv.shape)

        # Visibility test: all corners inside the image and in front of camera.
        W, H = scene.intrinsics.image_size
        visible = (pts_cam[:, 2] > 0.05).all() and (uv[:, 0] > 5).all() and (uv[:, 0] < W - 5).all() and (uv[:, 1] > 5).all() and (uv[:, 1] < H - 5).all()

        image = _render_scene(scene, model, T_ee_to_base, row, uv, visible, T_board_to_camera)
        cv2.imwrite(str(images_dir / f"{idx}.jpg"), image)
        poses.append(
            {
                "index": idx,
                "joint_angles_deg": row,
                "T_ee_to_base": T_ee_to_base,
                "T_board_to_camera_gt": T_board_to_camera,
                "T_board_to_base_gt": T_board_now,
                "visible": visible,
            }
        )

    np.savez(
        out_dir / "intrinsics.npz",
        K=scene.intrinsics.K,
        distortion=scene.intrinsics.distortion,
        image_size=np.array(scene.intrinsics.image_size),
        board_pattern_size=np.array(scene.board.pattern_size),
        board_square_size=scene.board.square_size,
    )
    np.savetxt(out_dir / "joint_angles.txt", joint_angles_deg, fmt="%.4f")
    np.savez(
        out_dir / "ground_truth.npz",
        T_camera_to_base=scene.T_camera_to_base,
        T_board_to_end_effector=scene.T_board_to_end_effector,
        world_points=world_pts,
    )
    np.savez(
        out_dir / "camextrinsics_gt.npz",
        T_board_to_camera=np.stack([p["T_board_to_camera_gt"] for p in poses]),
        T_board_to_base=np.stack([p["T_board_to_base_gt"] for p in poses]),
        T_ee_to_base=np.stack([p["T_ee_to_base"] for p in poses]),
        visible=np.array([p["visible"] for p in poses]),
    )
    return {"scene": scene, "poses": poses}


# --------------------------------------------------------------------------- #
# Very simple software renderer (not photorealistic, only for pipeline tests)
# --------------------------------------------------------------------------- #
def _render_scene(scene: SyntheticScene, model: RobotModel, T_ee_to_base, joint_angles_deg, uv, visible, T_board_to_camera=None) -> np.ndarray:
    W, H = scene.intrinsics.image_size
    image = np.full((H, W, 3), 40, np.uint8)

    # Ground plane grid to give some texture.
    for x in np.arange(-0.6, 0.7, 0.1):
        _draw_line(image, scene.intrinsics, T_ee_to_base, [[x, -0.6, 0], [x, 0.6, 0]], (55, 55, 55))
    for y in np.arange(-0.6, 0.7, 0.1):
        _draw_line(image, scene.intrinsics, T_ee_to_base, [[-0.6, y, 0], [0.6, y, 0]], (55, 55, 55))

    # Robot skeleton (links drawn between joint frames).
    frames = list(model.link_frames(np.deg2rad(joint_angles_deg)).values())
    for a, b in zip(frames[:-1], frames[1:]):
        _draw_line(image, scene.intrinsics, np.eye(4), [a[:3, 3], b[:3, 3]], (200, 200, 200), thickness=12)

    if not visible or T_board_to_camera is None:
        return image

    # Checkerboard squares: project the *exact* square corners (inner corner
    # grid extended by one square on every side, no extrapolation error).
    # Inner corners (r = 1..rows, c = 1..cols) must coincide with
    # Checkerboard.world_points(), i.e. ((c - 1) * s, (r - 1) * s).
    cols, rows = scene.board.pattern_size
    square = scene.board.square_size
    grid_world = np.zeros((rows + 2, cols + 2, 3), np.float64)
    for r in range(rows + 2):
        for c in range(cols + 2):
            grid_world[r, c, :2] = ((c - 1) * square, (r - 1) * square)
    grid = scene.intrinsics.project(transform_points(T_board_to_camera, grid_world.reshape(-1, 3))).reshape(grid_world.shape[:2] + (2,))

    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            quad = np.array([grid[r, c], grid[r, c + 1], grid[r + 1, c + 1], grid[r + 1, c]], np.int32)
            # Checkerboard parity must match the convention used by
            # cv2.findChessboardCorners for the object points in
            # Checkerboard.world_points() (verified per view in the self-test).
            color = (20, 20, 20) if (r + c) % 2 == 0 else (235, 235, 235)
            cv2.fillConvexPoly(image, quad, color, lineType=cv2.LINE_AA)
    return image


def _draw_line(image, intrinsics, T_base_to_camera, points_base, color, thickness=2):
    pts_cam = transform_points(T_base_to_camera, np.asarray(points_base, float))
    if (pts_cam[:, 2] < 0.05).any():
        return
    uv = intrinsics.project(pts_cam).astype(int)
    cv2.line(image, tuple(uv[0]), tuple(uv[1]), color, thickness, cv2.LINE_AA)


def pose_report(T_camera_to_base: np.ndarray) -> str:
    R = T_camera_to_base[:3, :3]
    t = T_camera_to_base[:3, 3]
    roll, pitch, yaw = euler_zyx_deg(R)
    return (
        f"translation [m] = {np.round(t, 4).tolist()}\n"
        f"rpy (ZYX) [deg] = [{roll:.2f}, {pitch:.2f}, {yaw:.2f}]"
    )

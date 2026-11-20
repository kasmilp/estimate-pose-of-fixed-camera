"""Nonlinear reprojection refinement of a hand-eye calibration.

The linear Park-Martin / OpenCV solvers only use the rigid-motion
constraints.  Once we have an initial estimate, we can minimize the actual
quantity we care about: the reprojection error of every checkerboard corner
in every view,

    r_i = project( X^-1 B_i Y world_points ) - detected_pixels_i

with X = T_camera_to_base and Y = T_board_to_end_effector.  A dozen
parameters (6 + 6) are optimized with scipy's Levenberg-Marquardt/TRF
solver.  This typically halves the translation error on noisy data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import CameraIntrinsics
from .se3 import invert, make_transform, matrix_to_rotvec, transform_points


@dataclass
class RefinementResult:
    T_camera_to_base: np.ndarray
    T_board_to_end_effector: np.ndarray
    rmse_before_px: float
    rmse_after_px: float
    nfev: int
    improved: bool


def refine_hand_eye(
    board_to_camera: np.ndarray,
    end_effector_to_base: np.ndarray,
    image_points: np.ndarray,
    world_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_camera_to_base: np.ndarray,
    T_board_to_end_effector: np.ndarray,
    max_nfev: int = 100,
) -> RefinementResult:
    """Refine ``T_camera_to_base`` and ``T_board_to_end_effector``.

    Parameters
    ----------
    board_to_camera : (N, 4, 4)
        Camera extrinsics (camera_T_board) for every view.
    end_effector_to_base : (N, 4, 4)
        Forward kinematics (base_T_gripper) for every view.
    image_points : (N, P, 2)
        Detected checkerboard corners.  They must be in *undistorted*
        (ideal pinhole) pixel coordinates, because the calibration pipeline
        undistorts the images before detection.
    world_points : (P, 3)
        Checkerboard object points in the board frame.
    intrinsics : CameraIntrinsics
        Intrinsics of the undistorted images (distortion is ignored here).
    T_camera_to_base, T_board_to_end_effector : (4, 4)
        Initial guesses from the linear hand-eye solution.
    """
    board_to_camera = np.asarray(board_to_camera, float)
    end_effector_to_base = np.asarray(end_effector_to_base, float)
    image_points = np.asarray(image_points, float).reshape(len(board_to_camera), -1, 2)
    world_points = np.asarray(world_points, float)

    # Detected corners are in ideal pixel coordinates -> zero distortion.
    ideal = CameraIntrinsics(
        K=np.asarray(intrinsics.K, float),
        distortion=np.zeros_like(np.asarray(intrinsics.distortion, float)),
        image_size=intrinsics.image_size,
    )

    def unpack(params: np.ndarray):
        X = make_transform(Rotation.from_rotvec(params[0:3]).as_matrix(), params[3:6])
        Y = make_transform(Rotation.from_rotvec(params[6:9]).as_matrix(), params[9:12])
        return X, Y

    def residuals(params: np.ndarray) -> np.ndarray:
        X, Y = unpack(params)
        X_inv = invert(X)
        out = []
        for B, uv in zip(end_effector_to_base, image_points):
            C = X_inv @ B @ Y
            pred = ideal.project(transform_points(C, world_points))
            out.append((pred - uv).ravel())
        return np.concatenate(out)

    def rmse(params: np.ndarray) -> float:
        r = residuals(params)
        return float(np.sqrt(np.mean(r**2)))

    x0 = np.concatenate(
        [
            matrix_to_rotvec(T_camera_to_base[:3, :3]),
            T_camera_to_base[:3, 3],
            matrix_to_rotvec(T_board_to_end_effector[:3, :3]),
            T_board_to_end_effector[:3, 3],
        ]
    )
    before = rmse(x0)
    sol = least_squares(
        residuals,
        x0,
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    after = rmse(sol.x)
    if not np.isfinite(after) or after > before:
        X, Y = unpack(x0)
        return RefinementResult(X, Y, before, before, int(sol.nfev), False)

    X, Y = unpack(sol.x)
    return RefinementResult(X, Y, before, after, int(sol.nfev), True)

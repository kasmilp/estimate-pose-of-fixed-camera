"""Nonlinear reprojection refinement of an eye-to-hand calibration.

The linear ``AX = XB`` solvers only use the rigid-motion constraints.  Once
we have an initial estimate we can minimize the quantity that actually
matters: the reprojection error of every checkerboard corner in every view.

For ``X = T_camera_to_base`` and ``Y = T_board_to_end_effector``::

    C_i(pred) = X^-1 B_i Y
    r_i       = project(C_i(pred) @ world_points) - detected_pixels_i

Only 12 parameters (6 for X, 6 for Y) are optimized.  On noisy data this
typically halves the translation error of the linear solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .camera import CameraIntrinsics
from .transforms import (
    inverse,
    make_transform,
    rotation_to_rotvec,
    rotvec_to_rotation,
    transform_points,
)

__all__ = ["RefinementResult", "refine_hand_eye"]


@dataclass
class RefinementResult:
    T_camera_to_base: np.ndarray
    T_board_to_end_effector: np.ndarray
    rmse_before_px: float
    rmse_after_px: float
    nfev: int
    success: bool
    improved: bool


def refine_hand_eye(
    T_board_to_camera: np.ndarray,
    T_end_effector_to_base: np.ndarray,
    image_points: np.ndarray,
    world_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_camera_to_base: np.ndarray,
    T_board_to_end_effector: np.ndarray,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int = 200,
) -> RefinementResult:
    """Refine ``T_camera_to_base`` and ``T_board_to_end_effector``.

    ``image_points`` must be measured in **undistorted** pixel coordinates.
    """
    C = np.asarray(T_board_to_camera, dtype=float)
    B = np.asarray(T_end_effector_to_base, dtype=float)
    uv_all = np.asarray(image_points, dtype=float)
    world = np.asarray(world_points, dtype=float).reshape(-1, 3)
    if uv_all.ndim != 3 or uv_all.shape[2] != 2:
        raise ValueError("image_points must have shape (N, P, 2)")
    if len(uv_all) != len(C) or len(B) != len(C):
        raise ValueError("one image-point array per robot pose is required")

    ideal = intrinsics.without_distortion()

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = make_transform(rotvec_to_rotation(params[0:3]), params[3:6])
        Y = make_transform(rotvec_to_rotation(params[6:9]), params[9:12])
        return X, Y

    def residuals(params: np.ndarray) -> np.ndarray:
        X, Y = unpack(params)
        X_inv = inverse(X)
        chunks = []
        for B_i, uv_i in zip(B, uv_all):
            C_pred = X_inv @ B_i @ Y
            predicted = ideal.project(transform_points(C_pred, world), apply_distortion=False)
            chunks.append((predicted - uv_i).ravel())
        return np.concatenate(chunks)

    def rmse(params: np.ndarray) -> float:
        r = residuals(params)
        return float(np.sqrt(np.mean(r**2)))

    x0 = np.concatenate(
        [
            rotation_to_rotvec(T_camera_to_base[:3, :3]),
            np.asarray(T_camera_to_base[:3, 3], dtype=float),
            rotation_to_rotvec(T_board_to_end_effector[:3, :3]),
            np.asarray(T_board_to_end_effector[:3, 3], dtype=float),
        ]
    )

    before = rmse(x0)
    solution = least_squares(
        residuals,
        x0,
        method="trf",
        x_scale="jac",
        loss=loss,
        f_scale=f_scale,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    after = rmse(solution.x)

    if not np.isfinite(after) or after > before:
        X, Y = unpack(x0)
        return RefinementResult(X, Y, before, before, int(solution.nfev), bool(solution.success), False)

    X, Y = unpack(solution.x)
    return RefinementResult(X, Y, before, after, int(solution.nfev), bool(solution.success), True)

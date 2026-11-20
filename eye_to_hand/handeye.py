"""Eye-to-hand hand-eye calibration: solve ``A X = X B``.

Setup (the one shown in the project diagram)
--------------------------------------------
* the camera is mounted on a fixed beam (eye-to-hand / stationary camera),
* a checkerboard is rigidly attached to the robot end effector,
* for every robot pose ``i`` we know

  ``C_i = T_board_to_camera``  (camera extrinsics, from PnP)
  ``B_i = T_end_effector_to_base`` (forward kinematics)

The unknown is ``X = T_camera_to_base``.  For every pair ``(i, j)`` the
rigid-motion constraint is

    (B_j B_i^-1) X = X (C_j C_i^-1)

i.e. ``A X = X B`` with ``A = B_j B_i^-1`` and ``B = C_j C_i^-1``.  The pair
direction must be the same on both sides; mixing ``C_i^-1 C_j`` silently
solves the eye-in-hand problem and returns a roughly 180 degree wrong pose.

The default solver is the log-rotation / SVD method of Park & Martin.  The
OpenCV solvers are available for comparison; note that OpenCV's eye-to-hand
input convention requires ``gripper2base = B_i^-1`` and
``target2cam = C_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .transforms import (
    average_rotation,
    inverse,
    make_transform,
    rotation_angle,
    rotation_to_rotvec,
)

__all__ = [
    "HandEyeSolution",
    "METHODS",
    "DEFAULT_METHOD",
    "estimate_camera_robot_transform",
    "solve_eye_to_hand",
    "evaluate_solution",
    "solve_all_methods",
    "method_agreement",
]

DEFAULT_METHOD = "park"

# OpenCV's calibrateHandEye implementations, exposed under clear names.
OPENCV_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "opencv_park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

METHODS = (DEFAULT_METHOD, *OPENCV_METHODS)


@dataclass
class HandEyeSolution:
    """Result of an eye-to-hand calibration."""

    T_camera_to_base: np.ndarray
    method: str
    T_board_to_end_effector_per_pose: np.ndarray
    T_board_to_end_effector: np.ndarray
    rotation_spread_deg: float
    translation_spread_mm: float
    pair_rotation_residual_deg: np.ndarray
    pair_translation_residual_mm: np.ndarray

    @property
    def T_base_to_camera(self) -> np.ndarray:
        return inverse(self.T_camera_to_base)


def solve_eye_to_hand(
    T_board_to_camera: np.ndarray,
    T_end_effector_to_base: np.ndarray,
    method: str = DEFAULT_METHOD,
) -> HandEyeSolution:
    """Estimate the fixed camera pose ``base_T_camera``.

    Parameters
    ----------
    T_board_to_camera : (N, 4, 4)
        Camera extrinsics ``camera_T_board`` for every view.
    T_end_effector_to_base : (N, 4, 4)
        Forward kinematics ``base_T_end_effector`` for every view.
    method : str
        ``park`` (default, dependency-light numpy implementation) or one of
        ``tsai``, ``opencv_park``, ``horaud``, ``andreff``, ``daniilidis``.
    """
    C = np.asarray(T_board_to_camera, dtype=float)
    B = np.asarray(T_end_effector_to_base, dtype=float)
    if C.ndim != 3 or C.shape[1:] != (4, 4):
        raise ValueError("T_board_to_camera must have shape (N, 4, 4)")
    if B.shape != C.shape:
        raise ValueError("T_end_effector_to_base must have the same shape as T_board_to_camera")
    if len(C) < 3:
        raise ValueError("at least 3 robot poses are required (>= 10 recommended)")

    method_key = method.lower()
    if method_key not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {METHODS}")

    if method_key == DEFAULT_METHOD:
        X = _solve_park(B, C)
    else:
        X = _solve_opencv(B, C, OPENCV_METHODS[method_key])

    return evaluate_solution(X, C, B, method_key)


def estimate_camera_robot_transform(
    board_to_camera: np.ndarray,
    end_effector_to_base: np.ndarray,
    configuration: str = "stationary-camera",
    method: str = DEFAULT_METHOD,
) -> HandEyeSolution:
    """MATLAB-compatible wrapper (``stationary-camera`` only).

    ``configuration="moving-camera"`` is rejected explicitly rather than
    silently producing a wrong answer.
    """
    if configuration.lower() not in ("stationary-camera", "eye-to-hand", "eyetohand"):
        raise ValueError("this project implements the stationary-camera / eye-to-hand case")
    return solve_eye_to_hand(board_to_camera, end_effector_to_base, method)


# ---------------------------------------------------------------------------
# Park-Martin solver (returns base_T_camera)
# ---------------------------------------------------------------------------
def _solve_park(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Park-Martin log-rotation solution of ``A X = X B``.

    For every pair ``i < j``:

    * ``A = B_j B_i^-1``, ``B_rel = C_j C_i^-1``
    * rotation: ``log(R_A) = R_X log(R_Brel)``
    * translation: ``(R_A - I) t_X = R_X t_Brel - t_A``
    """
    log_A: list[np.ndarray] = []
    log_B: list[np.ndarray] = []
    t_A: list[np.ndarray] = []
    t_B: list[np.ndarray] = []

    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            A = B[j] @ inverse(B[i])
            B_rel = C[j] @ inverse(C[i])
            log_A.append(rotation_to_rotvec(A[:3, :3]))
            log_B.append(rotation_to_rotvec(B_rel[:3, :3]))
            t_A.append(A[:3, 3])
            t_B.append(B_rel[:3, 3])

    alpha = np.asarray(log_A)
    beta = np.asarray(log_B)
    ta = np.asarray(t_A)
    tb = np.asarray(t_B)

    # Orthogonal Procrustes on the log rotations: alpha ~= R_X beta.
    M = alpha.T @ beta
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R_X = U @ D @ Vt

    # Stack the translation equations.
    lhs = np.vstack([(B[j] @ inverse(B[i]))[:3, :3] - np.eye(3) for i in range(len(B)) for j in range(i + 1, len(B))])
    rhs = np.concatenate([R_X @ b - a for a, b in zip(ta, tb)])
    t_X, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    return make_transform(R_X, t_X)


# ---------------------------------------------------------------------------
# OpenCV solvers (also return base_T_camera)
# ---------------------------------------------------------------------------
def _solve_opencv(B: np.ndarray, C: np.ndarray, flag: int) -> np.ndarray:
    """OpenCV eye-to-hand solve with the documented input convention."""
    gripper_to_base = [inverse(T) for T in B]
    R_gripper2base = [T[:3, :3] for T in gripper_to_base]
    t_gripper2base = [T[:3, 3] for T in gripper_to_base]
    R_target2cam = [T[:3, :3] for T in C]
    t_target2cam = [T[:3, 3] for T in C]
    R, t = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=flag,
    )
    return make_transform(R, np.asarray(t).ravel())


# ---------------------------------------------------------------------------
# Consistency metrics
# ---------------------------------------------------------------------------
def evaluate_solution(
    T_camera_to_base: np.ndarray,
    T_board_to_camera: np.ndarray,
    T_end_effector_to_base: np.ndarray,
    method: str = "park",
    T_board_to_end_effector: np.ndarray | None = None,
) -> HandEyeSolution:
    """Package a camera pose together with its self-consistency metrics.

    ``T_board_to_end_effector`` can be supplied when a nonlinear refinement
    produced a better board pose than the chordal mean of the per-view poses.
    """
    X = np.asarray(T_camera_to_base, dtype=float)
    C = np.asarray(T_board_to_camera, dtype=float)
    B = np.asarray(T_end_effector_to_base, dtype=float)

    # Y = T_board_to_end_effector = B_i^-1 X C_i must be constant.
    Y_per_pose = np.stack([inverse(B_i) @ X @ C_i for B_i, C_i in zip(B, C)])
    mean_R = average_rotation(Y_per_pose[:, :3, :3])
    mean_t = Y_per_pose[:, :3, 3].mean(axis=0)
    Y = make_transform(mean_R, mean_t)
    if T_board_to_end_effector is not None:
        Y = np.asarray(T_board_to_end_effector, dtype=float)

    rot_spread = max(rotation_angle(mean_R.T @ R_i) for R_i in Y_per_pose[:, :3, :3])
    trans_spread = float(np.max(np.linalg.norm(Y_per_pose[:, :3, 3] - mean_t, axis=1)))

    pair_rot: list[float] = []
    pair_trans: list[float] = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            A = B[j] @ inverse(B[i])
            B_rel = C[j] @ inverse(C[i])
            E = inverse(X @ B_rel) @ (A @ X)
            pair_rot.append(rotation_angle(E[:3, :3]))
            pair_trans.append(float(np.linalg.norm(E[:3, 3])))

    return HandEyeSolution(
        T_camera_to_base=X,
        method=method,
        T_board_to_end_effector_per_pose=Y_per_pose,
        T_board_to_end_effector=Y,
        rotation_spread_deg=float(np.degrees(rot_spread)),
        translation_spread_mm=float(trans_spread * 1000.0),
        pair_rotation_residual_deg=np.degrees(np.asarray(pair_rot)),
        pair_translation_residual_mm=np.asarray(pair_trans) * 1000.0,
    )


def solve_all_methods(
    T_board_to_camera: np.ndarray,
    T_end_effector_to_base: np.ndarray,
) -> dict[str, HandEyeSolution]:
    """Run every available solver; useful for a robustness cross-check."""
    results: dict[str, HandEyeSolution] = {}
    for name in METHODS:
        try:
            results[name] = solve_eye_to_hand(T_board_to_camera, T_end_effector_to_base, name)
        except (cv2.error, np.linalg.LinAlgError) as exc:  # pragma: no cover - data dependent
            print(f"[warn] method {name} failed: {exc}")
    return results


def method_agreement(
    results: dict[str, HandEyeSolution],
    reference: str = DEFAULT_METHOD,
) -> list[tuple[str, float, float]]:
    """Rotation (deg) / translation (mm) difference of every method vs reference."""
    from .transforms import transform_error

    ref = results[reference].T_camera_to_base
    rows = []
    for name, solution in results.items():
        rot, trans = transform_error(ref, solution.T_camera_to_base)
        rows.append((name, float(np.degrees(rot)), float(trans * 1000.0)))
    return rows

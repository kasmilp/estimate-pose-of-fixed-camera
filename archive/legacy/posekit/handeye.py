"""Hand-eye (eye-to-hand) solver: estimate the fixed camera pose w.r.t. the base.

Python equivalent of
    TCameraToBase = estimateCameraRobotTransform(camextrinsics, TEndEffectorToBase, "stationary-camera");

Math used
---------
Stationary camera + board on the gripper (eye-to-hand) gives, for every
pair of views (i, j), the rigid-motion (AX = XB) constraint

    (B_j B_i^-1) X = X (C_j C_i^-1)

with  B = T_end_effector_to_base   (forward kinematics)
      C = T_board_to_camera        (camera extrinsics)
      X = T_camera_to_base         (what we want)

Both sides use the *same* pair direction (j after i). Mixing
``B_j B_i^-1`` with ``C_i^-1 C_j`` would solve the eye-in-hand problem
and return a 180 deg-wrong camera pose.

The default solver is a dependency-light Park–Martin log-rotation solution
(``method="park_numpy"``) that returns X directly, so no input/output
convention ambiguity can silently produce a 180 deg-wrong answer. The OpenCV
methods (``cv2.calibrateHandEye``) are still available for comparison.

Conventions used here (verified in ``selftest_synthetic.py``):
  * inputs  : ``board_to_camera`` = camera_T_board, ``end_effector_to_base`` = base_T_gripper
  * output  : ``T_camera_to_base`` = base_T_camera, ``T_base_to_camera`` = camera_T_base
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .se3 import invert, make_transform, matrix_to_rotvec, rotation_angle, transform_difference

# OpenCV hand-eye methods that are numerically robust for eye-to-hand.
# (ANDREFF / DANIILIDIS are known to be less reliable on real data.)
OPENCV_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}
METHODS = {"park_numpy": None, **OPENCV_METHODS}
DEFAULT_METHOD = "park_numpy"


@dataclass
class HandEyeResult:
    T_camera_to_base: np.ndarray
    T_base_to_camera: np.ndarray
    method: str
    # Per-pose consistency: for each pose, B^-1 @ X @ C should be constant
    # (= pose of the board in the gripper frame, T_board_to_gripper).
    T_board_to_gripper_per_pose: np.ndarray
    T_board_to_gripper: np.ndarray  # chordal mean of the per-pose values
    rotation_spread_deg: float
    translation_spread_mm: float


def estimate_camera_robot_transform(
    board_to_camera: np.ndarray,
    end_effector_to_base: np.ndarray,
    configuration: str = "stationary-camera",
    method: str = DEFAULT_METHOD,
) -> HandEyeResult:
    """Estimate the fixed-camera pose relative to the robot base.

    Parameters
    ----------
    board_to_camera : (N, 4, 4)
        Camera extrinsics per view (``camextrinsics`` in MATLAB), camera_T_board.
    end_effector_to_base : (N, 4, 4)
        Forward kinematics per view (``TEndEffectorToBase`` in MATLAB), base_T_gripper.
    configuration : {"stationary-camera", "moving-camera"}
        ``stationary-camera`` == eye-to-hand (the MathWorks example).
    method : str
        ``park_numpy`` (default, recommended) or an OpenCV method:
        ``tsai``, ``park``, ``horaud``, ``andreff``, ``daniilidis``.
    """
    board_to_camera = np.asarray(board_to_camera, dtype=float)
    end_effector_to_base = np.asarray(end_effector_to_base, dtype=float)
    if board_to_camera.shape != end_effector_to_base.shape:
        raise ValueError("camextrinsics and TEndEffectorToBase must have the same shape (N, 4, 4)")
    if len(board_to_camera) < 3:
        raise ValueError("at least 3 poses are required (>= 15 recommended)")

    moving = configuration.lower() in ("moving-camera", "eye-in-hand", "eyeinhand")
    if not moving and configuration.lower() not in ("stationary-camera", "eye-to-hand", "eyetohand"):
        raise ValueError("configuration must be 'stationary-camera' or 'moving-camera'")

    method_key = method.lower()
    if method_key not in METHODS:
        raise ValueError(f"unknown method '{method}', choose from {sorted(METHODS)}")

    if method_key == "park_numpy":
        T_camera_to_base = _solve_eye_to_hand_park(end_effector_to_base, board_to_camera)
    else:
        T_camera_to_base = _solve_eye_to_hand_opencv(end_effector_to_base, board_to_camera, OPENCV_METHODS[method_key])

    if moving:
        T_camera_to_base = invert(T_camera_to_base)

    return hand_eye_result(T_camera_to_base, board_to_camera, end_effector_to_base, method_key)


def hand_eye_result(
    T_camera_to_base: np.ndarray,
    board_to_camera: np.ndarray,
    end_effector_to_base: np.ndarray,
    method: str = "refined",
) -> HandEyeResult:
    """Package a camera pose + its per-pose consistency metrics."""
    T_camera_to_base = np.asarray(T_camera_to_base, float)
    # Consistency check: T_board_to_gripper = B^-1 @ X @ C must be constant.
    board_to_gripper = np.stack([invert(B) @ (T_camera_to_base @ C) for B, C in zip(end_effector_to_base, board_to_camera)])
    rotations = board_to_gripper[:, :3, :3]
    mean_R = average_rotation(rotations)
    mean_t = board_to_gripper[:, :3, 3].mean(axis=0)
    rot_errs = np.array([rotation_angle(mean_R.T @ R) for R in rotations])
    trans_errs = np.linalg.norm(board_to_gripper[:, :3, 3] - mean_t, axis=1)

    return HandEyeResult(
        T_camera_to_base=T_camera_to_base,
        T_base_to_camera=invert(T_camera_to_base),
        method=method,
        T_board_to_gripper_per_pose=board_to_gripper,
        T_board_to_gripper=make_transform(mean_R, mean_t),
        rotation_spread_deg=float(np.degrees(rot_errs.max())),
        translation_spread_mm=float(trans_errs.max() * 1000.0),
    )


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def _solve_eye_to_hand_park(end_effector_to_base: np.ndarray, board_to_camera: np.ndarray) -> np.ndarray:
    """Park–Martin hand-eye via the log of rotations (returns base_T_camera).

    For pairs (i, j):  A = B_j B_i^-1,  Brel = C_j C_i^-1,  solve  A X = X Brel.
    (This is the eye-to-hand pair form: the target moves with the gripper.)

    Park & Martin (1994) linear formulation:
      log(R_A) = R_X log(R_Brel)   ->  R_X from M = sum log(R_A) log(R_Brel)^T
      (R_A - I) t_X = R_X t_Brel - t_A   (stacked over pairs, least squares)
    """
    n = len(end_effector_to_base)
    log_A, log_B, t_A, t_B = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            A = end_effector_to_base[j] @ invert(end_effector_to_base[i])
            Brel = board_to_camera[j] @ invert(board_to_camera[i])
            log_A.append(matrix_to_rotvec(A[:3, :3]))
            log_B.append(matrix_to_rotvec(Brel[:3, :3]))
            t_A.append(A[:3, 3])
            t_B.append(Brel[:3, 3])
    log_A, log_B = np.array(log_A), np.array(log_B)
    t_A, t_B = np.array(t_A), np.array(t_B)

    M = log_A.T @ log_B
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R_X = U @ D @ Vt

    # (R_A - I) t_X = R_X t_B - t_A, stacked over all pairs.
    lhs = np.vstack([_rodrigues(la) - np.eye(3) for la in log_A])
    rhs = np.concatenate([R_X @ tb - ta for tb, ta in zip(t_B, t_A)])
    t_X, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    return make_transform(R_X, t_X)


def _solve_eye_to_hand_opencv(end_effector_to_base: np.ndarray, board_to_camera: np.ndarray, flag: int) -> np.ndarray:
    """OpenCV solver. Returns base_T_camera for the inputs documented above.

    For the eye-to-hand configuration OpenCV expects the *inverse* gripper
    poses, ``gripper2base = gripper_T_base`` and ``target2cam = camera_T_board``.
    The returned transform is ``base_T_camera`` (verified against ground truth
    in ``selftest_synthetic.py``).
    """
    gripper_to_base = [invert(T) for T in end_effector_to_base]
    R_gripper2base = [T[:3, :3] for T in gripper_to_base]
    t_gripper2base = [T[:3, 3] for T in gripper_to_base]
    R_target2cam = [T[:3, :3] for T in board_to_camera]
    t_target2cam = [T[:3, 3] for T in board_to_camera]
    R, t = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=flag)
    return make_transform(R, t.ravel())


def solve_all_methods(board_to_camera, end_effector_to_base, configuration="stationary-camera") -> dict[str, HandEyeResult]:
    """Run every solver (numpy + OpenCV) so their agreement can be inspected."""
    results = {}
    for name in METHODS:
        try:
            results[name] = estimate_camera_robot_transform(board_to_camera, end_effector_to_base, configuration, name)
        except cv2.error as exc:  # pragma: no cover - depends on data
            print(f"[warn] method {name} failed: {exc}")
    return results


def method_spread(results: dict[str, HandEyeResult], reference: str = DEFAULT_METHOD) -> list[tuple[str, float, float]]:
    """Rotation (deg) / translation (mm) difference of each method vs reference."""
    ref = results[reference].T_camera_to_base
    out = []
    for name, res in results.items():
        d_rot, d_trans = transform_difference(ref, res.T_camera_to_base)
        out.append((name, np.degrees(d_rot), d_trans * 1000.0))
    return out


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rotvec).as_matrix()


def average_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (SVD projection)."""
    M = rotations.sum(axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

"""Rigid-body transforms (SO(3) / SE(3)) used throughout the project.

A homogeneous transform ``T_a_to_b`` maps points expressed in frame ``a``
into frame ``b``::

    p_b = T_a_to_b @ p_a

All transforms are 4x4 ``numpy`` arrays.  Rotations are 3x3 matrices.
The helpers in this module are deliberately small and explicit so that the
frame direction is always visible in the name of a variable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "rpy_to_rotation",
    "rotation_to_rpy",
    "make_transform",
    "inverse",
    "transform_points",
    "rotation_angle",
    "transform_error",
    "rotvec_to_rotation",
    "rotation_to_rotvec",
    "rotation_to_quaternion",
    "quaternion_to_rotation",
    "average_rotation",
    "project_to_so3",
    "backproject_pixels",
    "triangulate_rays",
]


# ---------------------------------------------------------------------------
# Rotation constructors
# ---------------------------------------------------------------------------
def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_rotation(rpy: Sequence[float]) -> np.ndarray:
    """URDF convention: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = (float(v) for v in rpy)
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def rotation_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_to_rotation`, returned as ``[roll, pitch, yaw]``.

    The result is in radians.  A gimbal-lock safe fallback is used when
    ``|pitch|`` is close to 90 degrees.
    """
    R = np.asarray(R, dtype=float)
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # pitch = +/- 90 deg
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Transform constructors and basic operations
# ---------------------------------------------------------------------------
def make_transform(rotation: np.ndarray | None = None, translation: Sequence[float] | None = None) -> np.ndarray:
    """Build a 4x4 homogeneous transform."""
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    """Fast inverse of a rigid transform."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(N, 3)`` array of points."""
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# Rotation error / conversion
# ---------------------------------------------------------------------------
def rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation angle in radians (numerically stable near zero)."""
    R = np.asarray(R, dtype=float)
    # scipy uses a quaternion internally, which stays accurate for tiny angles.
    return float(Rotation.from_matrix(R).magnitude())


def transform_error(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """Return ``(rotation_error_rad, translation_error_m)`` between transforms."""
    D = inverse(T1) @ np.asarray(T2, dtype=float)
    return rotation_angle(D[:3, :3]), float(np.linalg.norm(D[:3, 3]))


def rotvec_to_rotation(rotvec: Sequence[float]) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float).reshape(3)).as_matrix()


def rotation_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return a quaternion in ``(w, x, y, z)`` order."""
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return np.array([w, x, y, z])


def quaternion_to_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Build a rotation matrix from a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto SO(3) (nearest rotation in Frobenius norm)."""
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def average_rotation(rotations: Iterable[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices."""
    stack = np.asarray(list(rotations), dtype=float)
    if stack.ndim != 3 or stack.shape[1:] != (3, 3):
        raise ValueError("expected a sequence of 3x3 rotation matrices")
    return project_to_so3(stack.sum(axis=0))


# ---------------------------------------------------------------------------
# Camera geometry helpers (kept here to avoid a circular import)
# ---------------------------------------------------------------------------
def backproject_pixels(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert ``(N, 2)`` pixels to unit rays in the camera frame."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    rays = homogeneous @ np.linalg.inv(np.asarray(K, dtype=float)).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def triangulate_rays(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray:
    """Closest point to two 3D rays (least squares).

    Used to check that a checkerboard corner seen from the camera and the
    same physical corner predicted by forward kinematics intersect.
    """
    oa = np.asarray(origin_a, dtype=float).reshape(3)
    da = np.asarray(direction_a, dtype=float).reshape(3)
    ob = np.asarray(origin_b, dtype=float).reshape(3)
    db = np.asarray(direction_b, dtype=float).reshape(3)
    A = np.column_stack([da, -db])
    st, *_ = np.linalg.lstsq(A, ob - oa, rcond=None)
    return oa + st[0] * da

"""Small SE(3) / SO(3) toolkit used across the project.

All transforms are 4x4 homogeneous matrices (numpy arrays) unless stated
otherwise. A transform ``T_a_to_b`` maps points expressed in frame *a* into
frame *b* (same convention as the MathWorks ``rigidtform3d`` example).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def rotx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def roty(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def make_transform(rotation: np.ndarray | None = None, translation=None) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --------------------------------------------------------------------------- #
# Rotation utilities
# --------------------------------------------------------------------------- #
def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix in radians."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def transform_difference(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """Return (rotation error [rad], translation error [m]) between transforms."""
    D = invert(T1) @ T2
    return rotation_angle(D[:3, :3]), float(np.linalg.norm(D[:3, 3]))


def rotvec_to_matrix(rvec) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion as (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw (degrees) for R = Rz @ Ry @ Rx."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


# --------------------------------------------------------------------------- #
# Points
# --------------------------------------------------------------------------- #
def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ T[:3, :3].T + T[:3, 3]


def triangulate_points(T_a_to_b: np.ndarray, T_c_to_b: np.ndarray, p_a, p_c) -> np.ndarray:
    """Least-squares intersection of two rays expressed in different frames.

    Used to check that the same physical point seen from the end-effector
    frame (checkerboard corner) and from the camera frame (back-projected
    pixel ray) coincides after applying the estimated transforms.
    """
    p_a = np.asarray(p_a, dtype=float)
    d_a = T_a_to_b[:3, :3] @ p_a
    o_a = T_a_to_b[:3, 3]
    p_c = np.asarray(p_c, dtype=float)
    d_c = T_c_to_b[:3, :3] @ p_c
    o_c = T_c_to_b[:3, 3]

    # closest point to two lines: solve [d_a, -d_c] [s t]^T = o_c - o_a
    A = np.column_stack([d_a, -d_c])
    s, *_ = np.linalg.lstsq(A, o_c - o_a, rcond=None)
    return o_a + s[0] * d_a


def backproject_pixel(uv, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) -> unit ray in the camera (OpenCV) frame."""
    K_inv = np.linalg.inv(K)
    hom = K_inv @ np.array([uv[0], uv[1], 1.0])
    return hom / np.linalg.norm(hom)

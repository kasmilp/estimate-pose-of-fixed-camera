"""Forward kinematics wrappers: robot model + end-effector poses.

Python equivalent of:
    kinova = loadrobot("kinovaGen3", DataFormat="row");
    TEndEffectorToBase(r) = getTransform(kinova, deg2rad(jointAngles(r,:)), "EndEffector_Link");
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .urdf import UrdfKinematics

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "data" / "kinova_gen3_7dof.urdf"
END_EFFECTOR_LINK = "EndEffector_Link"
BASE_LINK = "base_link"


class RobotModel:
    def __init__(self, urdf_path: str | Path = DEFAULT_URDF, end_effector_link: str = END_EFFECTOR_LINK):
        self.kinematics = UrdfKinematics(urdf_path)
        self.end_effector_link = end_effector_link

    @property
    def n_dof(self) -> int:
        return self.kinematics.n_dof

    def fk(self, q_rad) -> np.ndarray:
        """Pose of the end effector in the base frame (4x4)."""
        return self.kinematics.get_transform(q_rad, self.end_effector_link, BASE_LINK)

    def fk_many(self, joint_angles_deg) -> np.ndarray:
        """Stack of end-effector poses for an (N, n_dof) array in degrees."""
        joint_angles_deg = np.atleast_2d(np.asarray(joint_angles_deg, dtype=float))
        return np.stack([self.fk(np.deg2rad(row)) for row in joint_angles_deg])

    def link_frames(self, q_rad) -> dict[str, np.ndarray]:
        return self.kinematics.all_link_frames(q_rad)


# --------------------------------------------------------------------------- #
# MATLAB helpers kept 1:1 for a smooth port
# --------------------------------------------------------------------------- #
def deg2rad(value):
    return np.deg2rad(value)


def get_transform(model: RobotModel, jointconfig_rad, body: str = END_EFFECTOR_LINK) -> np.ndarray:
    """Drop-in replacement for ``getTransform``."""
    return model.kinematics.get_transform(jointconfig_rad, body, BASE_LINK)

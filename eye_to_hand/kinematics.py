"""Minimal URDF parser and forward kinematics.

This replaces ``loadrobot("kinovaGen3")`` + ``getTransform(robot, q,
"EndEffector_Link")`` from the MATLAB example.  Only the kinematic tree is
needed, so meshes and inertial data in the URDF are ignored.

The bundled model is the Kinova Gen3 7-DoF (Kortex ``GEN3_URDF_V12``) with
``base_link`` as the root and ``EndEffector_Link`` as the tool frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree as ET

import numpy as np

from .transforms import (
    inverse,
    make_transform,
    rotvec_to_rotation,
    rpy_to_rotation,
)

__all__ = ["Joint", "UrdfRobot", "RobotModel", "get_transform"]

MOVABLE_TYPES = {"revolute", "continuous", "prismatic"}
DEFAULT_URDF = Path(__file__).resolve().parent / "assets" / "kinova_gen3_7dof.urdf"
END_EFFECTOR_LINK = "EndEffector_Link"
BASE_LINK = "base_link"


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # parent -> child at q = 0
    axis: np.ndarray  # unit axis in the child/joint frame
    kind: str

    @property
    def movable(self) -> bool:
        return self.kind in MOVABLE_TYPES

    def motion(self, value: float) -> np.ndarray:
        """Joint motion transform for a revolute (rad) or prismatic (m) joint."""
        if self.kind == "prismatic":
            T = np.eye(4)
            T[:3, 3] = self.axis * float(value)
            return T
        return make_transform(rotvec_to_rotation(self.axis * float(value)))


class UrdfRobot:
    """Forward kinematics for a serial / tree-structured URDF."""

    def __init__(self, urdf_path: str | Path = DEFAULT_URDF, root_link: str | None = None):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        root = ET.parse(self.urdf_path).getroot()
        self.joints: list[Joint] = []
        for element in root.findall("joint"):
            origin_el = element.find("origin")
            xyz = _parse_floats(origin_el.get("xyz", "0 0 0")) if origin_el is not None else np.zeros(3)
            rpy = _parse_floats(origin_el.get("rpy", "0 0 0")) if origin_el is not None else np.zeros(3)
            axis_el = element.find("axis")
            axis = _parse_floats(axis_el.get("xyz", "0 0 1")) if axis_el is not None else np.array([0.0, 0.0, 1.0])
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            self.joints.append(
                Joint(
                    name=element.get("name", ""),
                    parent=element.find("parent").get("link"),  # type: ignore[union-attr]
                    child=element.find("child").get("link"),  # type: ignore[union-attr]
                    origin=make_transform(rpy_to_rotation(rpy), xyz),
                    axis=axis,
                    kind=element.get("type", "fixed"),
                )
            )

        self.links = [link.get("name") for link in root.findall("link")]
        self._parent_joint = {joint.child: joint for joint in self.joints}
        self._children: dict[str, list[Joint]] = {}
        for joint in self.joints:
            self._children.setdefault(joint.parent, []).append(joint)

        if root_link is None:
            children = {joint.child for joint in self.joints}
            roots = [link for link in self.links if link not in children]
            if len(roots) != 1:
                raise ValueError(f"could not determine a unique root link: {roots}")
            root_link = roots[0]
        self.root_link = root_link

        self.movable_joint_names = [joint.name for joint in self.joints if joint.movable]
        self.n_dof = len(self.movable_joint_names)
        self._joint_index = {name: i for i, name in enumerate(self.movable_joint_names)}

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def chain_to(self, link: str) -> list[Joint]:
        """Joints from the root link down to ``link`` (root first)."""
        chain: list[Joint] = []
        while link != self.root_link:
            if link not in self._parent_joint:
                raise KeyError(f"link '{link}' is not connected to root '{self.root_link}'")
            joint = self._parent_joint[link]
            chain.append(joint)
            link = joint.parent
        return chain[::-1]

    def _value(self, q, joint_name: str) -> float:
        if isinstance(q, Mapping):
            return float(q[joint_name])
        return float(np.asarray(q, dtype=float).ravel()[self._joint_index[joint_name]])

    def _fk_to_link(self, q, link: str) -> np.ndarray:
        if link == self.root_link:
            return np.eye(4)
        T = np.eye(4)
        for joint in self.chain_to(link):
            T = T @ joint.origin
            if joint.movable:
                T = T @ joint.motion(self._value(q, joint.name))
        return T

    def get_transform(self, q, target_link: str, base_link: str | None = None) -> np.ndarray:
        """Pose of ``target_link`` expressed in ``base_link`` (default: root)."""
        base_link = base_link or self.root_link
        T_root_target = self._fk_to_link(q, target_link)
        T_root_base = self._fk_to_link(q, base_link)
        return inverse(T_root_base) @ T_root_target

    def all_link_frames(self, q) -> dict[str, np.ndarray]:
        """Pose of every link expressed in the root frame."""
        frames: dict[str, np.ndarray] = {self.root_link: np.eye(4)}

        def visit(parent: str) -> None:
            for joint in self._children.get(parent, []):
                T = frames[parent] @ joint.origin
                if joint.movable:
                    T = T @ joint.motion(self._value(q, joint.name))
                frames[joint.child] = T
                visit(joint.child)

        visit(self.root_link)
        return frames


class RobotModel:
    """Convenience wrapper around :class:`UrdfRobot` for a robot arm."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF,
        base_link: str = BASE_LINK,
        end_effector_link: str = END_EFFECTOR_LINK,
    ):
        self.robot = UrdfRobot(urdf_path)
        self.base_link = base_link
        self.end_effector_link = end_effector_link

    @property
    def n_dof(self) -> int:
        return self.robot.n_dof

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot.movable_joint_names)

    def fk(self, q_rad) -> np.ndarray:
        """End-effector pose in the base frame from joint angles in radians."""
        return self.robot.get_transform(q_rad, self.end_effector_link, self.base_link)

    def fk_deg(self, q_deg) -> np.ndarray:
        """End-effector pose in the base frame from joint angles in degrees."""
        return self.fk(np.deg2rad(q_deg))

    def fk_many(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        """Stack of end-effector poses for an ``(N, n_dof)`` degree array."""
        angles = np.atleast_2d(np.asarray(joint_angles_deg, dtype=float))
        if angles.shape[1] != self.n_dof:
            raise ValueError(f"expected {self.n_dof} joint angles per row, got {angles.shape[1]}")
        return np.stack([self.fk_deg(row) for row in angles])

    def link_frames(self, q_rad) -> dict[str, np.ndarray]:
        return self.robot.all_link_frames(q_rad)


def get_transform(model: RobotModel, joint_config_rad, body: str = END_EFFECTOR_LINK) -> np.ndarray:
    """MATLAB-style ``getTransform`` helper."""
    return model.robot.get_transform(joint_config_rad, body, model.base_link)


def _parse_floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=float)

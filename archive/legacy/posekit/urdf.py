"""Minimal URDF parser + forward kinematics.

This is the Python counterpart of ``loadrobot("kinovaGen3")`` and
``getTransform(robot, q, "EndEffector_Link")`` from the MathWorks example.
Only the kinematic information (joint origin, axis, parent/child) is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .se3 import make_transform, rpy_to_matrix

MOVABLE_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4, parent -> child at zero configuration
    axis: np.ndarray  # unit axis expressed in the child frame (URDF)
    kind: str


class UrdfKinematics:
    """Forward kinematics for a serial URDF chain."""

    def __init__(self, urdf_path: str | Path, root_link: str | None = None):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()

        self.joints: list[Joint] = []
        for j in root.findall("joint"):
            origin_el = j.find("origin")
            xyz = _floats(origin_el.get("xyz", "0 0 0")) if origin_el is not None else np.zeros(3)
            rpy = _floats(origin_el.get("rpy", "0 0 0")) if origin_el is not None else np.zeros(3)
            axis_el = j.find("axis")
            axis = _floats(axis_el.get("xyz", "0 0 1")) if axis_el is not None else np.array([0.0, 0.0, 1.0])
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            self.joints.append(
                Joint(
                    name=j.get("name"),
                    parent=j.find("parent").get("link"),
                    child=j.find("child").get("link"),
                    origin=make_transform(rpy_to_matrix(rpy), xyz),
                    axis=axis,
                    kind=j.get("type"),
                )
            )

        self.links = [l.get("name") for l in root.findall("link")]
        self._by_child = {j.child: j for j in self.joints}

        if root_link is None:
            children = {j.child for j in self.joints}
            root_link = next(l for l in self.links if l not in children)
        self.root_link = root_link

        # Ordered serial chain: list of (joint) from root to every link.
        self.movable_joint_names = [j.name for j in self.joints if j.kind in MOVABLE_TYPES]
        self.n_dof = len(self.movable_joint_names)

    # ------------------------------------------------------------------ #
    def chain_to(self, link: str) -> list[Joint]:
        """Joints from the root link down to ``link`` (root first)."""
        chain: list[Joint] = []
        while link != self.root_link:
            joint = self._by_child[link]
            chain.append(joint)
            link = joint.parent
        return chain[::-1]

    def get_transform(self, q, target_link: str, base_link: str | None = None) -> np.ndarray:
        """Pose of ``target_link`` expressed in ``base_link`` (default: root)."""
        base_link = base_link or self.root_link
        chain = self.chain_to(target_link)
        base_chain = self.chain_to(base_link)
        # base_link is assumed to be an ancestor of target_link: skip the
        # shared prefix so that base_link becomes the reference frame.
        T = np.eye(4)
        start = len(base_chain)
        q = np.asarray(q, dtype=float).ravel()
        index = {name: i for i, name in enumerate(self.movable_joint_names)}
        for joint in chain[start:]:
            T = T @ joint.origin
            if joint.kind in MOVABLE_TYPES:
                angle = q[index[joint.name]]
                T = T @ _joint_motion(joint, angle)
        return T

    def all_link_frames(self, q) -> dict[str, np.ndarray]:
        """Pose of every link expressed in the root frame (for plotting)."""
        frames = {self.root_link: np.eye(4)}
        index = {name: i for i, name in enumerate(self.movable_joint_names)}
        q = np.asarray(q, dtype=float).ravel()
        for joint in self.joints:
            if joint.parent not in frames:
                continue
            T = frames[joint.parent] @ joint.origin
            if joint.kind in MOVABLE_TYPES:
                T = T @ _joint_motion(joint, q[index[joint.name]])
            frames[joint.child] = T
        return frames


def _joint_motion(joint: Joint, angle: float) -> np.ndarray:
    from .se3 import rotvec_to_matrix

    if joint.kind == "prismatic":
        T = np.eye(4)
        T[:3, 3] = joint.axis * angle
        return T
    return make_transform(rotvec_to_matrix(joint.axis * angle))


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=float)

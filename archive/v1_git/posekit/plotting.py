"""Plotting helpers (matplotlib) used by the pipeline steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .se3 import transform_points


def plot_montage(image_paths, out_path: str | Path, cols: int = 5, title: str = "Calibration Images"):
    image_paths = list(image_paths)
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, image_paths):
        ax.imshow(_read_bgr(path))
        ax.set_title(Path(path).name, fontsize=8)
        ax.axis("off")
    for ax in axes[len(image_paths) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_detections(image_paths, extrinsics, out_path: str | Path, max_images: int = 6, intrinsics=None):
    """Overlay detected corners + reprojected board frame.

    Detections are made on the *undistorted* image, so the displayed image is
    undistorted too when ``intrinsics`` is provided.
    """
    import cv2

    image_paths = list(image_paths)[:max_images]
    cols = 3
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, path, ext in zip(axes, image_paths, extrinsics):
        image = cv2.imread(str(path))
        if intrinsics is not None and np.any(intrinsics.distortion):
            image = cv2.undistort(image, intrinsics.K, intrinsics.distortion)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if ext is not None:
            ax.scatter(ext.image_points[:, 0], ext.image_points[:, 1], s=6, c="lime")
            ax.set_title(f"{Path(path).name}  RMSE={ext.reprojection_rmse:.2f}px", fontsize=8)
        else:
            ax.set_title(f"{Path(path).name}  (no detection)", fontsize=8)
        ax.axis("off")
    for ax in axes[len(image_paths) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_setup(T_camera_to_base, T_ee_to_base_list, world_points, out_path: str | Path, board_pose=None):
    """3D view: robot end-effector frames, camera and checkerboard in the base frame."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for T in T_ee_to_base_list:
        p = T[:3, 3]
        ax.scatter(*p, s=12, c="tab:blue", alpha=0.6)
        _quiver_frame(ax, T, scale=0.06, colors=("r", "g", "b"))

    if board_pose is not None:
        pts = transform_points(board_pose, world_points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c="k", marker="*", label="board points")

    _quiver_frame(ax, T_camera_to_base, scale=0.15, colors=("m", "y", "c"))
    cam = T_camera_to_base[:3, 3]
    ax.scatter(*cam, s=80, c="magenta", marker="^", label="fixed camera")
    ax.scatter(0, 0, 0, s=80, c="k", marker="s", label="robot base")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Estimated camera frame with respect to the robot base")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=15, azim=-60)
    _equal_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_errors(errors_deg, errors_mm, out_path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(np.arange(len(errors_deg)), errors_deg, color="tab:orange")
    axes[0].set_title("Rotation error [deg]")
    axes[0].set_xlabel("pair / pose index")
    axes[1].bar(np.arange(len(errors_mm)), errors_mm, color="tab:green")
    axes[1].set_title("Translation error [mm]")
    axes[1].set_xlabel("pair / pose index")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _quiver_frame(ax, T, scale, colors):
    o = T[:3, 3]
    for i, c in enumerate(colors):
        ax.quiver(*o, *(T[:3, i] * scale), color=c, linewidth=1.5)


def _equal_axes(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _read_bgr(path):
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

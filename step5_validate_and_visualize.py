"""Step 5 - Validation & visualisation.

* Left panel  : robot + checkerboard points resolved in the base frame
                (``TWorldToBase * worldpts`` in MATLAB).
* Right panel : the corresponding calibration image.
* Overlay     : detected corners (green circles) and the corners predicted
                from forward kinematics + the estimated hand-eye transform
                (red crosses).
* Metrics     : reprojection error of the *predicted* board pose and the
                intersection error between the FK ray and the camera ray.

The prediction is deliberately **not** the PnP board pose again: it uses

    camera_T_board_pred = T_camera_to_base^-1 * B_i * T_board_to_end_effector

so the residual measures the hand-eye + FK consistency of the whole chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from posekit.camera import CameraIntrinsics, undistort_image
from posekit.io import list_images
from posekit.plotting import _equal_axes, _quiver_frame, plot_setup
from posekit.robot import RobotModel
from posekit.se3 import backproject_pixel, invert, transform_points, triangulate_points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--view", type=int, default=0, help="index of the view used for the side-by-side plot")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    extr = np.load(data_dir / "camextrinsics.npz")
    res = np.load(data_dir / "result.npz")
    intrinsics = CameraIntrinsics.from_calibration_file(data_dir / "intrinsics.npz")
    joint_angles_deg = np.loadtxt(data_dir / "joint_angles.txt", ndmin=2)
    images = list_images(data_dir / "images")

    T_camera_to_base = res["T_camera_to_base"]
    T_board_to_end_effector = res["T_board_to_end_effector"]
    T_board_to_camera = extr["T_board_to_camera"]
    T_ee_to_base = extr["T_ee_to_base"]
    image_points = extr["image_points"]
    world_points = extr["world_points"]
    used = np.asarray(extr["used_indices"], dtype=int)

    if len(used) != len(T_board_to_camera):
        raise SystemExit("camextrinsics.npz is inconsistent: used_indices length mismatch")

    # Detections were made on undistorted images.
    ideal_intrinsics = CameraIntrinsics(
        K=intrinsics.K,
        distortion=np.zeros_like(intrinsics.distortion),
        image_size=intrinsics.image_size,
    )

    # ---- 1. camera / robot / board in the base frame ---------------------- #
    plot_setup(
        T_camera_to_base,
        T_ee_to_base,
        world_points,
        reports_dir / "step5_camera_frame_in_base.png",
        board_pose=res["T_board_to_base"],
    )

    # ---- 2. side-by-side: 3D board points vs calibration image ------------ #
    model = RobotModel()
    view = int(np.clip(args.view, 0, len(used) - 1))
    image_index = int(used[view])
    image_path = images[image_index]

    fig = plt.figure(figsize=(11, 4.5))
    ax3d = fig.add_subplot(121, projection="3d")
    frames = model.link_frames(np.deg2rad(joint_angles_deg[image_index]))
    pts = np.array([T[:3, 3] for T in frames.values()])
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", c="tab:gray", ms=3, lw=2)
    board_pts = transform_points(T_ee_to_base[view] @ T_board_to_end_effector, world_points)
    ax3d.scatter(board_pts[:, 0], board_pts[:, 1], board_pts[:, 2], c="k", s=8, marker="*")
    _quiver_frame(ax3d, T_ee_to_base[view], 0.08, ("r", "g", "b"))
    ax3d.set_title("Checkerboard points in the base frame")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    _equal_axes(ax3d)

    ax_img = fig.add_subplot(122)
    ax_img.imshow(cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Calibration image {image_path.name}")
    ax_img.axis("off")
    fig.tight_layout()
    fig.savefig(reports_dir / "step5_board_vs_image.png", dpi=110)
    plt.close(fig)

    # ---- 3. predict the board from FK + hand-eye and reproject ------------ #
    predicted_px = []
    errors_px = []
    for B, uv in zip(T_ee_to_base, image_points):
        C_pred = invert(T_camera_to_base) @ B @ T_board_to_end_effector
        proj = ideal_intrinsics.project(transform_points(C_pred, world_points))
        predicted_px.append(proj)
        errors_px.append(np.linalg.norm(proj - uv, axis=1))

    errors_px = np.array(errors_px)
    predicted_px = np.array(predicted_px)

    # PnP-only reprojection RMSE, for comparison (stored by step 3).
    pnp_rmse = np.asarray(extr["reprojection_rmse"], dtype=float)

    # ---- 4. ray intersection: FK ray vs camera ray ------------------------ #
    triang_err = []
    for B, uv in zip(T_ee_to_base, image_points):
        p_ee = transform_points(T_board_to_end_effector, world_points)  # corners in the EE frame
        p_base = transform_points(B, p_ee)
        for p, pix, p_true in zip(p_ee, uv, p_base):
            d_cam = backproject_pixel(pix, intrinsics.K)
            p_tri = triangulate_points(B, T_camera_to_base, p, d_cam)
            triang_err.append(np.linalg.norm(p_tri - p_true))
    triang_err = np.array(triang_err) * 1000.0

    print(f"PnP reprojection RMSE (step 3)      : mean {pnp_rmse.mean():.2f} px, max {pnp_rmse.max():.2f} px")
    print(f"FK + hand-eye predicted reprojection: mean {errors_px.mean():.2f} px, max {errors_px.max():.2f} px")
    print(f"ray intersection error              : mean {triang_err.mean():.2f} mm, max {triang_err.max():.2f} mm")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(errors_px.ravel(), bins=30, color="tab:blue")
    axes[0].set_title("FK + hand-eye reprojection error [px]")
    axes[1].hist(triang_err, bins=30, color="tab:red")
    axes[1].set_title("Ray intersection error [mm]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports_dir / "step5_error_histograms.png", dpi=110)
    plt.close(fig)

    # ---- 5. overlay for every used view ----------------------------------- #
    overlay_dir = reports_dir / "step5_overlay"
    overlay_dir.mkdir(exist_ok=True)
    used_set = set(int(v) for v in used)
    for idx, image_file in enumerate(images):
        if idx not in used_set:
            continue
        k = int(np.where(used == idx)[0][0])
        image = undistort_image(cv2.imread(str(image_file)), intrinsics)
        for det, pred in zip(image_points[k], predicted_px[k]):
            cv2.circle(image, tuple(np.round(det).astype(int)), 4, (0, 255, 0), 1)
            cv2.drawMarker(image, tuple(np.round(pred).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
        cv2.putText(
            image,
            f"mean {errors_px[k].mean():.2f} px",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(overlay_dir / image_file.name), image)

    # ---- 6. text report ---------------------------------------------------- #
    per_view = [
        f"  view {int(used[k]) + 1:>2}: PnP {pnp_rmse[k]:5.2f} px | predicted {errors_px[k].mean():5.2f} px (max {errors_px[k].max():5.2f})"
        for k in range(len(used))
    ]
    report = "\n".join(
        [
            "Validation",
            "==========",
            f"views used                          : {len(used)}",
            f"PnP reprojection RMSE               : mean {pnp_rmse.mean():.3f} px, max {pnp_rmse.max():.3f} px",
            f"FK + hand-eye predicted reprojection: mean {errors_px.mean():.3f} px, max {errors_px.max():.3f} px",
            f"ray intersection error              : mean {triang_err.mean():.3f} mm, max {triang_err.max():.3f} mm",
            "",
            "per view:",
            *per_view,
            "",
            "Interpretation: the PnP row only reflects corner noise.  The predicted",
            "row additionally contains the hand-eye and board-mounting error, so it",
            "is the meaningful end-to-end check.  A mean below ~1 px and a ray",
            "error of a few mm are typical; grazing views can be slightly worse.",
        ]
    )
    (reports_dir / "step5_validation.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nreports: {reports_dir}")


if __name__ == "__main__":
    main()

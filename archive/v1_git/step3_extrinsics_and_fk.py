"""Step 3 - Camera extrinsics (board -> camera) + end-effector poses (base <- EE).

Python equivalent of the two MATLAB blocks:

    for r = 1:numimgs
        [imgpts, boardsz] = detectCheckerboardPoints(undistortImage(...), PartialDetections=false);
        worldpts = patternWorldPoints("checkerboard", boardsz, checkerboardSquareSizeInMeters);
        camextrinsics(r) = estimateExtrinsics(imgpts, worldpts, intrinsics);
    end

    for r = 1:numimgs
        TEndEffectorToBase(r) = getTransform(kinova, deg2rad(jointAngles(r,:)), "EndEffector_Link");
    end

Outputs ``<data-dir>/camextrinsics.npz`` (T_board_to_camera, T_ee_to_base) and
a detection report image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from posekit.camera import CameraIntrinsics, Checkerboard, compute_camera_extrinsics
from posekit.io import list_images
from posekit.plotting import plot_detections
from posekit.robot import RobotModel
from posekit.se3 import transform_difference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--max-rmse", type=float, default=1.5, help="reject a view if its reprojection RMSE exceeds this (px)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = CameraIntrinsics.from_calibration_file(data_dir / "intrinsics.npz")
    meta = np.load(data_dir / "intrinsics.npz")
    board = Checkerboard(
        pattern_size=tuple(int(v) for v in meta["board_pattern_size"]),
        square_size=float(meta["board_square_size"]),
    )
    joint_angles_deg = np.loadtxt(data_dir / "joint_angles.txt", ndmin=2)

    image_paths = list_images(data_dir / "images")
    if len(image_paths) != len(joint_angles_deg):
        raise SystemExit(f"{len(image_paths)} images vs {len(joint_angles_deg)} joint rows")

    print(f"board: {board.pattern_size[0]}x{board.pattern_size[1]} inner corners, {board.square_size*1000:.1f} mm squares")
    print("Estimating camera extrinsics ...")
    extrinsics = compute_camera_extrinsics(image_paths, intrinsics, board)

    print("\nComputing end-effector poses with forward kinematics ...")
    model = RobotModel()
    T_ee_to_base_all = model.fk_many(joint_angles_deg)

    # Keep only the views where the board was fully detected and accurate.
    keep = [i for i, e in enumerate(extrinsics) if e is not None and e.reprojection_rmse <= args.max_rmse]
    dropped = [i for i, e in enumerate(extrinsics) if i not in keep]
    if dropped:
        print(f"\n[dropped] views {[i + 1 for i in dropped]} (not detected or RMSE > {args.max_rmse} px)")
    if len(keep) < 3:
        raise SystemExit("not enough valid views for hand-eye calibration")

    T_board_to_camera = np.stack([extrinsics[i].T_board_to_camera for i in keep])
    T_ee_to_base = T_ee_to_base_all[keep]

    np.savez(
        data_dir / "camextrinsics.npz",
        T_board_to_camera=T_board_to_camera,
        T_ee_to_base=T_ee_to_base,
        used_indices=np.array(keep),
        image_points=np.stack([extrinsics[i].image_points for i in keep]),
        world_points=board.world_points(),
        reprojection_rmse=np.array([extrinsics[i].reprojection_rmse for i in keep]),
    )

    rmse = np.array([extrinsics[i].reprojection_rmse for i in keep])
    print(f"\nused {len(keep)} views | reprojection RMSE: mean {rmse.mean():.3f} px, max {rmse.max():.3f} px")

    plot_detections(image_paths, extrinsics, reports_dir / "step3_detections.png", intrinsics=intrinsics)
    print(f"\nreport: {reports_dir / 'step3_detections.png'}")
    print(f"saved : {data_dir / 'camextrinsics.npz'}")


if __name__ == "__main__":
    main()

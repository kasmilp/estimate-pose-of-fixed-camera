"""Pipeline self-test on a synthetic dataset (ground truth available).

    python selftest_synthetic.py

Checks that the full chain
    intrinsics -> checkerboard detection -> solvePnP -> hand-eye -> refinement
recovers the known ``T_camera_to_base`` and reports every linear solver for
comparison.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from posekit.camera import CameraIntrinsics, Checkerboard, compute_camera_extrinsics
from posekit.handeye import solve_all_methods
from posekit.io import list_images
from posekit.refine import refine_hand_eye
from posekit.robot import RobotModel
from posekit.se3 import transform_difference, transform_points
from posekit.synthetic import generate

TOL_ROT_DEG = 1.0
TOL_TRANS_MM = 5.0


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="eyetohand_"))
    print(f"synthetic dataset: {out_dir}")
    data = generate(out_dir, noise_px=0.3)
    scene = data["scene"]

    intrinsics = CameraIntrinsics.from_calibration_file(out_dir / "intrinsics.npz")
    board = Checkerboard()
    images = list_images(out_dir / "images")
    joint_angles_deg = np.loadtxt(out_dir / "joint_angles.txt", ndmin=2)

    extrinsics = compute_camera_extrinsics(images, intrinsics, board, verbose=False)
    keep = [i for i, e in enumerate(extrinsics) if e is not None and e.reprojection_rmse < 1.5]
    print(f"valid views: {len(keep)} / {len(images)}")
    if len(keep) < 3:
        raise SystemExit("FAIL: not enough valid views")

    T_board_to_camera = np.stack([extrinsics[i].T_board_to_camera for i in keep])
    image_points = np.stack([extrinsics[i].image_points for i in keep])
    T_ee_to_base = RobotModel().fk_many(joint_angles_deg)[keep]

    results = solve_all_methods(T_board_to_camera, T_ee_to_base)
    gt = scene.T_camera_to_base

    print(f"\n{'method':<18} {'rot err [deg]':>13} {'trans err [mm]':>15}")
    for name, res in results.items():
        d_rot, d_trans = transform_difference(gt, res.T_camera_to_base)
        print(f"{name:<18} {np.degrees(d_rot):13.3f} {d_trans * 1000:15.2f}")

    # The reprojection refinement is the recommended final step; the required
    # pass criterion is based on it (Tsai is known to be unstable when the
    # robot rotations are nearly parallel, as in this dataset).
    linear = results["park_numpy"]
    refined = refine_hand_eye(
        T_board_to_camera,
        T_ee_to_base,
        image_points,
        board.world_points(),
        intrinsics,
        linear.T_camera_to_base,
        linear.T_board_to_gripper,
    )
    d_rot, d_trans = transform_difference(gt, refined.T_camera_to_base)
    print(f"{'park_numpy+refine':<18} {np.degrees(d_rot):13.3f} {d_trans * 1000:15.2f}")
    print(f"  reprojection RMSE: {refined.rmse_before_px:.3f} -> {refined.rmse_after_px:.3f} px")
    print("  (Tsai/Andreff are ill-conditioned for this robot path; pass/fail uses park_numpy+refine.)")

    # Board points resolved in the base frame with the estimated camera pose must
    # coincide with the ground-truth board location (MATLAB: TWorldToBase * worldpts).
    board_err = []
    world_points = board.world_points()
    for C, B in zip(T_board_to_camera, T_ee_to_base):
        est = transform_points(refined.T_camera_to_base @ C, world_points)
        truth = transform_points(B @ scene.T_board_to_end_effector, world_points)
        board_err.append(np.linalg.norm(est - truth, axis=1).max())
    board_err = np.array(board_err) * 1000
    print(f"\nboard points in base frame: mean {board_err.mean():.2f} mm, max {board_err.max():.2f} mm")

    ok = np.degrees(d_rot) <= TOL_ROT_DEG and d_trans * 1000 <= TOL_TRANS_MM
    print(f"\nresult: {'PASS' if ok else 'FAIL'} (tolerance {TOL_ROT_DEG} deg / {TOL_TRANS_MM} mm)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

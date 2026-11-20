"""Step 4 - Solve the hand-eye problem.

Python equivalent of

    TCameraToBase = estimateCameraRobotTransform(camextrinsics, TEndEffectorToBase, "stationary-camera");
    TWorldToBase  = rigidtform3d(TCameraToBase.A * camextrinsics(end).A);
    worldpts3d    = transformPointsForward(TWorldToBase, [worldpts, zeros(size(worldpts,1),1)]);

Saves ``<data-dir>/result.npz`` with T_camera_to_base, the board pose in the
base frame and the checkerboard points resolved in the base frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from posekit.camera import CameraIntrinsics
from posekit.handeye import DEFAULT_METHOD, estimate_camera_robot_transform, hand_eye_result, method_spread, solve_all_methods
from posekit.plotting import plot_errors, plot_setup
from posekit.refine import refine_hand_eye
from posekit.se3 import euler_zyx_deg, matrix_to_quat, rotation_angle, transform_difference, transform_points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=["park_numpy", "tsai", "park", "horaud", "andreff", "daniilidis"])
    parser.add_argument("--configuration", default="stationary-camera", choices=["stationary-camera", "moving-camera"])
    parser.add_argument("--no-refine", action="store_true", help="skip the nonlinear reprojection refinement")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_dir / "camextrinsics.npz")
    T_board_to_camera = data["T_board_to_camera"]
    T_ee_to_base = data["T_ee_to_base"]
    image_points = data["image_points"]
    world_points = data["world_points"]
    intrinsics = CameraIntrinsics.from_calibration_file(data_dir / "intrinsics.npz")

    print(f"Solving eye-to-hand with '{args.method}' on {len(T_board_to_camera)} poses ...")
    result = estimate_camera_robot_transform(T_board_to_camera, T_ee_to_base, args.configuration, args.method)

    refine_lines = []
    if not args.no_refine and args.configuration != "stationary-camera":
        print("\n[note] nonlinear refinement is only implemented for 'stationary-camera' -> skipped")
    elif not args.no_refine:
        refined = refine_hand_eye(
            T_board_to_camera,
            T_ee_to_base,
            image_points,
            world_points,
            intrinsics,
            result.T_camera_to_base,
            result.T_board_to_gripper,
        )
        refine_lines = [
            f"reprojection RMSE before refinement : {refined.rmse_before_px:.3f} px",
            f"reprojection RMSE after refinement  : {refined.rmse_after_px:.3f} px ({refined.nfev} iterations)",
        ]
        print("\nNonlinear refinement:")
        print("\n".join("  " + line for line in refine_lines))
        if refined.improved:
            result = hand_eye_result(refined.T_camera_to_base, T_board_to_camera, T_ee_to_base, args.method + "+refine")
            result.T_board_to_gripper = refined.T_board_to_end_effector
        else:
            refine_lines.append("refinement did not improve the reprojection error -> linear result kept")

    T_camera_to_base = result.T_camera_to_base
    R, t = T_camera_to_base[:3, :3], T_camera_to_base[:3, 3]
    roll, pitch, yaw = euler_zyx_deg(R)
    quat_wxyz = matrix_to_quat(R)

    # Board points in the base frame using the last view (MATLAB uses camextrinsics(end)).
    T_board_to_base = T_camera_to_base @ T_board_to_camera[-1]
    world_points_base = transform_points(T_board_to_base, world_points)

    print("\nT_camera_to_base =")
    print(np.array2string(T_camera_to_base, precision=5, suppress_small=True))
    print(f"\ntranslation [m]   : {np.round(t, 4).tolist()}")
    print(f"rotation rpy [deg]: roll {roll:.2f}, pitch {pitch:.2f}, yaw {yaw:.2f}")
    print(f"quaternion (wxyz) : {np.round(quat_wxyz, 5).tolist()}")
    print(f"\nconsistency across poses (board pose in the gripper frame should be constant):")
    print(f"  rotation spread    : {result.rotation_spread_deg:.2f} deg")
    print(f"  translation spread : {result.translation_spread_mm:.2f} mm")

    # Cross-check the OpenCV methods against each other.
    all_results = solve_all_methods(T_board_to_camera, T_ee_to_base, args.configuration)
    print(f"\nmethod agreement (vs linear '{args.method}'):")
    spread_lines = []
    for name, d_rot, d_trans in method_spread(all_results, args.method):
        line = f"  {name:<11} d_rot {d_rot:6.2f} deg   d_trans {d_trans:7.2f} mm"
        print(line)
        spread_lines.append(line)

    # Ground truth, when a synthetic dataset was used.
    gt_lines = []
    gt_path = data_dir / "ground_truth.npz"
    if gt_path.exists():
        gt = np.load(gt_path)["T_camera_to_base"]
        d_rot, d_trans = transform_difference(gt, T_camera_to_base)
        gt_lines = [
            f"ground-truth rotation error    : {np.degrees(d_rot):.3f} deg",
            f"ground-truth translation error : {d_trans * 1000:.2f} mm",
        ]
        print("\n" + "\n".join(gt_lines))

    np.savez(
        data_dir / "result.npz",
        T_camera_to_base=T_camera_to_base,
        T_base_to_camera=result.T_base_to_camera,
        T_board_to_base=T_board_to_base,
        T_board_to_end_effector=result.T_board_to_gripper,
        world_points_base=world_points_base,
        rpy_zyx_deg=np.array([roll, pitch, yaw]),
        quaternion_wxyz=quat_wxyz,
        method=np.array(result.method),
    )

    # Plots: camera + robot + board resolved in the base frame.
    plot_setup(T_camera_to_base, T_ee_to_base, world_points, reports_dir / "step4_camera_in_base.png", board_pose=T_board_to_base)
    pair_rot, pair_trans = _pairwise_residuals(T_board_to_camera, T_ee_to_base, T_camera_to_base)
    plot_errors(np.degrees(pair_rot), pair_trans * 1000, reports_dir / "step4_pairwise_errors.png")

    report = "\n".join(
        [
            "Hand-eye calibration result (eye-to-hand)",
            "=========================================",
            f"poses used        : {len(T_board_to_camera)}",
            f"method            : {result.method}",
            f"configuration     : {args.configuration}",
            *refine_lines,
            "",
            "T_camera_to_base =",
            np.array2string(T_camera_to_base, precision=6, suppress_small=True),
            "",
            f"translation [m]   : {np.round(t, 5).tolist()}",
            f"rpy (ZYX) [deg]   : {np.round([roll, pitch, yaw], 3).tolist()}",
            f"quaternion (wxyz) : {np.round(quat_wxyz, 6).tolist()}",
            "",
            f"rotation spread    : {result.rotation_spread_deg:.3f} deg",
            f"translation spread : {result.translation_spread_mm:.3f} mm",
            "",
            f"method agreement (vs linear '{args.method}'):",
            *spread_lines,
            *(["", *gt_lines] if gt_lines else []),
        ]
    )
    (reports_dir / "step4_result.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nreports: {reports_dir / 'step4_camera_in_base.png'}, {reports_dir / 'step4_result.txt'}")
    print(f"saved  : {data_dir / 'result.npz'}")


def _pairwise_residuals(T_board_to_camera, T_ee_to_base, T_camera_to_base):
    """Relative-motion residuals for the eye-to-hand form A X = X B.

    A = B_j B_i^-1 and B = C_j C_i^-1 (same pair direction).  The residual
    transform E = (X B)^-1 (A X) is the identity for a perfect solution.
    """
    n = len(T_board_to_camera)
    rot, trans = [], []
    X = T_camera_to_base
    for i in range(n):
        for j in range(i + 1, n):
            A = T_ee_to_base[j] @ np.linalg.inv(T_ee_to_base[i])
            B = T_board_to_camera[j] @ np.linalg.inv(T_board_to_camera[i])
            E = np.linalg.inv(X @ B) @ (A @ X)
            rot.append(rotation_angle(E[:3, :3]))
            trans.append(np.linalg.norm(E[:3, 3]))
    return np.array(rot), np.array(trans)


if __name__ == "__main__":
    main()

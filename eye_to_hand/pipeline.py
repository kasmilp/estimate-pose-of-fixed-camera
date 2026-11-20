"""End-to-end eye-to-hand calibration pipeline.

The pipeline is deliberately linear and easy to inspect:

1. detect the checkerboard in every image and solve ``T_board_to_camera``,
2. compute ``T_end_effector_to_base`` with URDF forward kinematics,
3. solve the linear ``A X = X B`` hand-eye problem,
4. optionally refine ``X`` and ``T_board_to_end_effector`` by minimizing
   the checkerboard reprojection error,
5. validate (reprojection, pose spread, ray intersection, ground truth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import BoardPose, CameraIntrinsics, Checkerboard, detect_checkerboard, estimate_board_pose
from .dataset import CalibrationDataset
from .handeye import (
    DEFAULT_METHOD,
    HandEyeSolution,
    method_agreement,
    solve_all_methods,
    solve_eye_to_hand,
)
from .kinematics import RobotModel
from .refine import RefinementResult, refine_hand_eye
from .transforms import (
    backproject_pixels,
    inverse,
    rotation_to_quaternion,
    rotation_to_rpy,
    transform_error,
    transform_points,
    triangulate_rays,
)

__all__ = [
    "Observation",
    "CalibrationConfig",
    "CalibrationResult",
    "detect_observations",
    "calibrate",
    "save_result",
    "summary_text",
    "result_to_dict",
]


@dataclass
class Observation:
    index: int
    image_path: Path
    image_points: np.ndarray
    T_board_to_camera: np.ndarray
    pnp_rmse_px: float


@dataclass
class CalibrationConfig:
    method: str = DEFAULT_METHOD
    refine: bool = True
    max_pnp_rmse_px: float = 1.5
    min_views: int = 3
    loss: str = "linear"
    f_scale: float = 1.0
    compare_methods: bool = True


@dataclass
class CalibrationResult:
    T_camera_to_base: np.ndarray
    T_base_to_camera: np.ndarray
    T_board_to_end_effector: np.ndarray
    method: str
    observations: list[Observation]
    used_indices: np.ndarray
    total_views: int
    joint_angles_deg_used: np.ndarray
    T_board_to_camera: np.ndarray
    T_ee_to_base: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    intrinsics: CameraIntrinsics
    board: Checkerboard
    pnp_rmse_px: np.ndarray
    per_view_reprojection_rmse_px: np.ndarray
    reprojection_rmse_px: float
    rotation_spread_deg: float
    translation_spread_mm: float
    pair_rotation_residual_deg: np.ndarray
    pair_translation_residual_mm: np.ndarray
    ray_error_mean_mm: float
    ray_error_max_mm: float
    linear_T_camera_to_base: np.ndarray
    refinement: RefinementResult | None = None
    ground_truth_error: tuple[float, float] | None = None
    method_agreement: list[tuple[str, float, float]] = field(default_factory=list)
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

    @property
    def views_used(self) -> int:
        return len(self.observations)

    @property
    def translation(self) -> np.ndarray:
        return self.T_camera_to_base[:3, 3]

    @property
    def rpy_zyx_deg(self) -> np.ndarray:
        return np.degrees(rotation_to_rpy(self.T_camera_to_base[:3, :3]))

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return rotation_to_quaternion(self.T_camera_to_base[:3, :3])


def detect_observations(
    dataset: CalibrationDataset,
    max_pnp_rmse_px: float = 1.5,
    verbose: bool = True,
) -> list[Observation]:
    """Detect the board and estimate camera extrinsics for every image."""
    ideal = dataset.intrinsics.without_distortion()
    observations: list[Observation] = []

    for index, path in enumerate(dataset.image_paths):
        image = cv2.imread(str(path))
        if image is None:
            if verbose:
                print(f"[warn] could not read {path.name}")
            continue
        image = dataset.intrinsics.undistort(image)
        corners = detect_checkerboard(image, dataset.board)
        if corners is None:
            if verbose:
                print(f"[drop] {path.name}: checkerboard not fully detected")
            continue
        pose: BoardPose = estimate_board_pose(corners, dataset.board, ideal)
        if pose.reprojection_rmse_px > max_pnp_rmse_px:
            if verbose:
                print(
                    f"[drop] {path.name}: PnP RMSE {pose.reprojection_rmse_px:.2f} px "
                    f"> {max_pnp_rmse_px:.2f} px"
                )
            continue
        observations.append(
            Observation(
                index=index,
                image_path=path,
                image_points=pose.image_points,
                T_board_to_camera=pose.T_board_to_camera,
                pnp_rmse_px=pose.reprojection_rmse_px,
            )
        )
        if verbose:
            print(f"[ok]   {path.name}: PnP RMSE {pose.reprojection_rmse_px:.2f} px")

    if verbose:
        print(f"[detect] kept {len(observations)}/{len(dataset.image_paths)} views")
    return observations


def calibrate(
    dataset: CalibrationDataset,
    config: CalibrationConfig | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Run detection -> linear hand-eye -> nonlinear refinement -> validation."""
    config = config or CalibrationConfig()
    dataset.validate()

    if verbose:
        print(f"[calibrate] board {dataset.board.columns}x{dataset.board.rows}, "
              f"{dataset.board.square_size * 1000:.1f} mm squares")
        print("[calibrate] detecting checkerboard ...")
    observations = detect_observations(dataset, config.max_pnp_rmse_px, verbose)
    if len(observations) < config.min_views:
        raise RuntimeError(f"only {len(observations)} usable views; need at least {config.min_views}")

    used = np.array([obs.index for obs in observations], dtype=int)
    C = np.stack([obs.T_board_to_camera for obs in observations])
    uv = np.stack([obs.image_points for obs in observations])
    pnp_rmse = np.array([obs.pnp_rmse_px for obs in observations], dtype=float)
    world = dataset.board.object_points()

    model = RobotModel()
    B = model.fk_many(dataset.joint_angles_deg[used])

    if verbose:
        print(f"[calibrate] solving AX = XB with '{config.method}' on {len(used)} poses ...")
    linear = solve_eye_to_hand(C, B, config.method)
    X = linear.T_camera_to_base
    Y = linear.T_board_to_end_effector
    method_label = config.method

    refinement: RefinementResult | None = None
    if config.refine:
        if verbose:
            print("[calibrate] nonlinear reprojection refinement ...")
        refinement = refine_hand_eye(
            C,
            B,
            uv,
            world,
            dataset.intrinsics,
            X,
            Y,
            loss=config.loss,
            f_scale=config.f_scale,
        )
        if verbose:
            print(
                f"[refine] reprojection RMSE {refinement.rmse_before_px:.3f} -> "
                f"{refinement.rmse_after_px:.3f} px "
                f"({'improved' if refinement.improved else 'kept linear result'})"
            )
        if refinement.improved:
            X = refinement.T_camera_to_base
            Y = refinement.T_board_to_end_effector
            method_label = f"{config.method}+refine"

    # Evaluate the final X/Y, including the per-pose consistency metrics.
    from .handeye import evaluate_solution

    solution: HandEyeSolution = evaluate_solution(X, C, B, method_label, T_board_to_end_effector=Y)

    per_view_rmse = _per_view_reprojection_rmse(X, Y, B, uv, world, dataset.intrinsics)
    ray_mean, ray_max = _ray_intersection_error(X, Y, B, uv, world, dataset.intrinsics)

    gt_error = None
    if dataset.ground_truth and "T_camera_to_base" in dataset.ground_truth:
        rot, trans = transform_error(dataset.ground_truth["T_camera_to_base"], X)
        gt_error = (float(np.degrees(rot)), float(trans * 1000.0))

    agreement: list[tuple[str, float, float]] = []
    if config.compare_methods:
        if verbose:
            print("[calibrate] comparing solvers ...")
        all_solutions = solve_all_methods(C, B)
        if all_solutions:
            reference = config.method if config.method in all_solutions else DEFAULT_METHOD
            if reference in all_solutions:
                agreement = method_agreement(all_solutions, reference)

    result = CalibrationResult(
        T_camera_to_base=X,
        T_base_to_camera=inverse(X),
        T_board_to_end_effector=Y,
        method=method_label,
        observations=observations,
        used_indices=used,
        total_views=len(dataset.image_paths),
        T_board_to_camera=C,
        T_ee_to_base=B,
        image_points=uv,
        world_points=world,
        intrinsics=dataset.intrinsics,
        board=dataset.board,
        pnp_rmse_px=pnp_rmse,
        per_view_reprojection_rmse_px=per_view_rmse,
        reprojection_rmse_px=float(np.sqrt(np.mean(per_view_rmse**2))),
        rotation_spread_deg=solution.rotation_spread_deg,
        translation_spread_mm=solution.translation_spread_mm,
        pair_rotation_residual_deg=solution.pair_rotation_residual_deg,
        pair_translation_residual_mm=solution.pair_translation_residual_mm,
        ray_error_mean_mm=ray_mean,
        ray_error_max_mm=ray_max,
        linear_T_camera_to_base=linear.T_camera_to_base,
        refinement=refinement,
        ground_truth_error=gt_error,
        method_agreement=agreement,
        config=config,
    )

    if verbose:
        print()
        print(summary_text(result))
    return result


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def _per_view_reprojection_rmse(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    uv: np.ndarray,
    world_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Reprojection error of the *predicted* board, not of the PnP pose."""
    ideal = intrinsics.without_distortion()
    X_inv = inverse(X)
    errors = []
    for B_i, uv_i in zip(B, uv):
        C_pred = X_inv @ B_i @ Y
        predicted = ideal.project(transform_points(C_pred, world_points), apply_distortion=False)
        errors.append(float(np.sqrt(np.mean(np.sum((predicted - uv_i) ** 2, axis=1)))))
    return np.asarray(errors, dtype=float)


def _ray_intersection_error(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    uv: np.ndarray,
    world_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Mean / max distance between the FK ray and the camera ray (mm)."""
    errors: list[float] = []
    points_ee = transform_points(Y, world_points)
    for B_i, uv_i in zip(B, uv):
        points_base = transform_points(B_i, points_ee)
        rays_cam = backproject_pixels(uv_i, intrinsics.K)
        for point_ee, point_base, ray_cam in zip(points_ee, points_base, rays_cam):
            point_triangulated = triangulate_rays(
                B_i[:3, 3],
                B_i[:3, :3] @ point_ee,
                X[:3, 3],
                X[:3, :3] @ ray_cam,
            )
            errors.append(float(np.linalg.norm(point_triangulated - point_base)))
    values = np.asarray(errors) * 1000.0
    return float(values.mean()), float(values.max())


# ---------------------------------------------------------------------------
# Reporting / persistence
# ---------------------------------------------------------------------------
def summary_text(result: CalibrationResult) -> str:
    X = result.T_camera_to_base
    t = X[:3, 3]
    rpy = result.rpy_zyx_deg
    quat = result.quaternion_wxyz
    lines = [
        "Eye-to-hand calibration",
        "=======================",
        f"views used         : {result.views_used}/{result.total_views}",
        f"method             : {result.method}",
        "",
        "T_camera_to_base =",
        np.array2string(X, precision=6, suppress_small=True),
        "",
        f"translation [m]    : {np.round(t, 5).tolist()}",
        f"rpy (ZYX) [deg]    : {np.round(rpy, 3).tolist()}",
        f"quaternion (wxyz)  : {np.round(quat, 6).tolist()}",
        "",
        f"PnP RMSE           : mean {result.pnp_rmse_px.mean():.3f} px, max {result.pnp_rmse_px.max():.3f} px",
        f"predicted RMSE     : mean {result.per_view_reprojection_rmse_px.mean():.3f} px, "
        f"max {result.per_view_reprojection_rmse_px.max():.3f} px",
        f"board spread       : {result.rotation_spread_deg:.3f} deg / {result.translation_spread_mm:.2f} mm",
        f"ray intersection   : mean {result.ray_error_mean_mm:.3f} mm, max {result.ray_error_max_mm:.3f} mm",
    ]
    if result.refinement is not None:
        lines.append(
            f"refinement         : {result.refinement.rmse_before_px:.3f} -> "
            f"{result.refinement.rmse_after_px:.3f} px ({result.refinement.nfev} evaluations)"
        )
    if result.ground_truth_error is not None:
        lines.append(
            f"ground truth error : {result.ground_truth_error[0]:.3f} deg / "
            f"{result.ground_truth_error[1]:.2f} mm"
        )
    if result.method_agreement:
        lines.append("")
        lines.append("solver agreement (vs reference):")
        for name, rot, trans in result.method_agreement:
            lines.append(f"  {name:<11} {rot:7.3f} deg  {trans:8.2f} mm")
    return "\n".join(lines)


def result_to_dict(result: CalibrationResult) -> dict:
    data = {
        "method": result.method,
        "views_used": result.views_used,
        "views_total": result.total_views,
        "used_indices_1based": (result.used_indices + 1).tolist(),
        "T_camera_to_base": result.T_camera_to_base.tolist(),
        "T_base_to_camera": result.T_base_to_camera.tolist(),
        "T_board_to_end_effector": result.T_board_to_end_effector.tolist(),
        "translation_m": result.translation.tolist(),
        "rpy_zyx_deg": result.rpy_zyx_deg.tolist(),
        "quaternion_wxyz": result.quaternion_wxyz.tolist(),
        "pnp_rmse_px": {
            "mean": float(result.pnp_rmse_px.mean()),
            "max": float(result.pnp_rmse_px.max()),
        },
        "reprojection_rmse_px": {
            "mean": float(result.per_view_reprojection_rmse_px.mean()),
            "max": float(result.per_view_reprojection_rmse_px.max()),
        },
        "board_pose_spread": {
            "rotation_deg": result.rotation_spread_deg,
            "translation_mm": result.translation_spread_mm,
        },
        "ray_intersection_mm": {
            "mean": result.ray_error_mean_mm,
            "max": result.ray_error_max_mm,
        },
        "board": result.board.to_dict(),
        "ground_truth_error": (
            None
            if result.ground_truth_error is None
            else {
                "rotation_deg": result.ground_truth_error[0],
                "translation_mm": result.ground_truth_error[1],
            }
        ),
        "refinement": (
            None
            if result.refinement is None
            else {
                "rmse_before_px": result.refinement.rmse_before_px,
                "rmse_after_px": result.refinement.rmse_after_px,
                "improved": result.refinement.improved,
                "nfev": result.refinement.nfev,
            }
        ),
        "solver_agreement": [
            {"method": name, "rotation_deg": rot, "translation_mm": trans}
            for name, rot, trans in result.method_agreement
        ],
        "config": {
            "method": result.config.method,
            "refine": result.config.refine,
            "max_pnp_rmse_px": result.config.max_pnp_rmse_px,
            "min_views": result.config.min_views,
            "loss": result.config.loss,
        },
    }
    return data


def save_result(result: CalibrationResult, out_dir: str | Path) -> dict[str, Path]:
    """Write ``result.npz``, ``result.json``, ``report.md`` and plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "result.npz",
        T_camera_to_base=result.T_camera_to_base,
        T_base_to_camera=result.T_base_to_camera,
        T_board_to_end_effector=result.T_board_to_end_effector,
        T_board_to_camera=result.T_board_to_camera,
        T_ee_to_base=result.T_ee_to_base,
        image_points=result.image_points,
        world_points=result.world_points,
        used_indices=result.used_indices,
        pnp_rmse_px=result.pnp_rmse_px,
        per_view_reprojection_rmse_px=result.per_view_reprojection_rmse_px,
        pair_rotation_residual_deg=result.pair_rotation_residual_deg,
        pair_translation_residual_mm=result.pair_translation_residual_mm,
        rotation_spread_deg=result.rotation_spread_deg,
        translation_spread_mm=result.translation_spread_mm,
        ray_error_mean_mm=result.ray_error_mean_mm,
        ray_error_max_mm=result.ray_error_max_mm,
        method=np.array(result.method),
        K=result.intrinsics.K,
        distortion=result.intrinsics.distortion,
        image_size=np.asarray(result.intrinsics.image_size, dtype=int),
        board_pattern_size=np.asarray(result.board.pattern_size, dtype=int),
        board_square_size=result.board.square_size,
    )

    payload = result_to_dict(result)
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary_text(result) + "\n", encoding="utf-8")

    from .reporting import save_plots, write_markdown_report

    plot_paths = save_plots(result, out_dir / "plots")
    report_path = write_markdown_report(result, out_dir / "report.md")

    paths = {
        "result_npz": out_dir / "result.npz",
        "result_json": out_dir / "result.json",
        "summary": out_dir / "summary.txt",
        "report": report_path,
        **plot_paths,
    }
    return paths

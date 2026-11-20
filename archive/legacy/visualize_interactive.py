"""Build an interactive 3D view of the eye-to-hand calibration.

    python visualize_interactive.py --data-dir data/calibration_synth --open

The generated ``reports/visualization.html`` is self-contained (Plotly is
embedded when it can be downloaded, otherwise the CDN is used).  It shows:

* the Kinova Gen3 skeleton at every calibration pose,
* the checkerboard attached to the end effector,
* the fixed camera, its optical axes and a frustum,
* the ground-truth camera pose when a synthetic dataset is available,
* the calibration image with detected / predicted corners.

Drag to orbit, scroll to zoom, right-drag to pan.  The slider, arrow keys and
the Play button animate the poses; clicking an end-effector or board marker
jumps to that pose.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from posekit.camera import CameraIntrinsics
from posekit.io import list_images
from posekit.robot import RobotModel
from posekit.se3 import euler_zyx_deg, invert, matrix_to_quat, transform_difference, transform_points

PLOTLY_VERSION = "2.35.2"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _segments(points_a, points_b) -> tuple[list, list, list]:
    """Flatten segment pairs into Plotly line arrays (None = break)."""
    xs, ys, zs = [], [], []
    for a, b in zip(points_a, points_b):
        xs.extend([float(a[0]), float(b[0]), None])
        ys.extend([float(a[1]), float(b[1]), None])
        zs.extend([float(a[2]), float(b[2]), None])
    return xs, ys, zs


def _grid_xyz(grid: np.ndarray) -> dict[str, list]:
    """Line segments along the rows and columns of a checkerboard grid."""
    rows, cols = grid.shape[:2]
    a, b = [], []
    for r in range(rows):
        for c in range(cols - 1):
            a.append(grid[r, c])
            b.append(grid[r, c + 1])
    for c in range(cols):
        for r in range(rows - 1):
            a.append(grid[r, c])
            b.append(grid[r + 1, c])
    x, y, z = _segments(a, b)
    return {"x": x, "y": y, "z": z}


def _box_mesh(transform: np.ndarray, half_w: float, half_h: float, depth: float):
    """Camera body as a small box behind the optical centre."""
    vertices = np.array(
        [
            [-half_w, -half_h, -depth],
            [half_w, -half_h, -depth],
            [half_w, half_h, -depth],
            [-half_w, half_h, -depth],
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        float,
    )
    points = transform_points(transform, vertices)
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [2, 3, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [1, 2, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return {
        "x": points[:, 0].tolist(),
        "y": points[:, 1].tolist(),
        "z": points[:, 2].tolist(),
        "i": i,
        "j": j,
        "k": k,
    }


def _plotly_script(use_cdn: bool = False) -> str:
    """Inline Plotly so the HTML also works without internet."""
    if use_cdn:
        return f'<script src="{PLOTLY_CDN}"></script>'
    cache = Path(tempfile.gettempdir()) / f"plotly-{PLOTLY_VERSION}.min.js"
    if not cache.exists():
        try:
            with urllib.request.urlopen(PLOTLY_CDN, timeout=20) as response:
                cache.write_bytes(response.read())
        except Exception:
            return f'<script src="{PLOTLY_CDN}"></script>'
    js = cache.read_text(encoding="utf-8", errors="replace")
    if "</script" in js.lower():
        js = js.replace("</script", "<\\/script").replace("</SCRIPT", "<\\/SCRIPT")
    return f"<script>{js}</script>"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration_synth")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--out", default=None, help="output HTML (default: <reports-dir>/visualization.html)")
    parser.add_argument("--cdn", action="store_true", help="use the Plotly CDN instead of embedding Plotly")
    parser.add_argument("--open", action="store_true", help="open the generated page in the default browser")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out) if args.out else reports_dir / "visualization.html"

    extr = np.load(data_dir / "camextrinsics.npz")
    res = np.load(data_dir / "result.npz")
    intr_data = np.load(data_dir / "intrinsics.npz")
    intr = CameraIntrinsics.from_calibration_file(data_dir / "intrinsics.npz")
    joints = np.loadtxt(data_dir / "joint_angles.txt", ndmin=2)
    images = list_images(data_dir / "images")

    cols, rows = (int(v) for v in intr_data["board_pattern_size"])
    square = float(intr_data["board_square_size"])

    T_ee_to_base = np.asarray(extr["T_ee_to_base"], float)
    image_points = np.asarray(extr["image_points"], float)
    world_points = np.asarray(extr["world_points"], float)
    used = np.asarray(extr["used_indices"], dtype=int)
    pnp_rmse = np.asarray(extr["reprojection_rmse"], float)
    T_camera_to_base = np.asarray(res["T_camera_to_base"], float)
    T_board_to_ee = np.asarray(res["T_board_to_end_effector"], float)

    ideal = CameraIntrinsics(
        K=intr.K,
        distortion=np.zeros_like(intr.distortion),
        image_size=intr.image_size,
    )
    model = RobotModel()
    n_views = len(used)

    grid_world = world_points.reshape(rows, cols, 3)
    plane_corners_w = np.array(
        [
            [-0.5 * square, -0.5 * square, 0.0],
            [(cols - 0.5) * square, -0.5 * square, 0.0],
            [(cols - 0.5) * square, (rows - 0.5) * square, 0.0],
            [-0.5 * square, (rows - 0.5) * square, 0.0],
        ],
        float,
    )
    board_centres = np.array([(T_ee_to_base[k] @ T_board_to_ee)[:3, 3] for k in range(n_views)])
    ee_positions = T_ee_to_base[:, :3, 3]

    # Camera frame / frustum --------------------------------------------------
    cam_origin = T_camera_to_base[:3, 3]
    depth = float(np.clip(np.median(np.linalg.norm(board_centres - cam_origin, axis=1)), 0.5, 2.0))
    W, H = intr.image_size
    K_inv = np.linalg.inv(intr.K)
    frustum_cam = (K_inv @ np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], float).T).T * depth
    frustum_base = transform_points(T_camera_to_base, frustum_cam)

    fx, fy, fz = _segments([cam_origin] * 4, frustum_base)
    ex, ey, ez = _segments(
        [frustum_base[0], frustum_base[1], frustum_base[2], frustum_base[3]],
        [frustum_base[1], frustum_base[2], frustum_base[3], frustum_base[0]],
    )
    frustum_xyz = {"x": fx + ex, "y": fy + ey, "z": fz + ez}
    camera_body = _box_mesh(T_camera_to_base, 0.048, 0.034, 0.11)

    # Ground truth ------------------------------------------------------------
    gt_path = data_dir / "ground_truth.npz"
    gt_rot = gt_trans = None
    gt_frustum = {"x": [], "y": [], "z": []}
    gt_marker = {"x": [], "y": [], "z": []}
    if gt_path.exists():
        T_gt = np.asarray(np.load(gt_path)["T_camera_to_base"], float)
        gt_rot, gt_trans = transform_difference(T_gt, T_camera_to_base)
        g_base = transform_points(T_gt, frustum_cam)
        gx, gy, gz = _segments([T_gt[:3, 3]] * 4, g_base)
        hx, hy, hz = _segments(
            [g_base[0], g_base[1], g_base[2], g_base[3]],
            [g_base[1], g_base[2], g_base[3], g_base[0]],
        )
        gt_frustum = {"x": gx + hx, "y": gy + hy, "z": gz + hz}
        gt_marker = {"x": [float(T_gt[0, 3])], "y": [float(T_gt[1, 3])], "z": [float(T_gt[2, 3])]}

    # All board outlines + poses ---------------------------------------------
    outline_a, outline_b = [], []
    poses = []
    pred_means, pred_maxs = [], []
    for k, image_idx in enumerate(used):
        B = T_ee_to_base[k]
        T_board_to_base = B @ T_board_to_ee
        board_pts = transform_points(T_board_to_base, world_points)
        plane_pts = transform_points(T_board_to_base, plane_corners_w)
        for i in range(4):
            outline_a.append(plane_pts[i])
            outline_b.append(plane_pts[(i + 1) % 4])

        frames = model.link_frames(np.deg2rad(joints[image_idx]))
        link_pts = []
        for T in frames.values():
            p = T[:3, 3]
            if not link_pts or np.linalg.norm(p - link_pts[-1]) > 1e-6:
                link_pts.append(p)

        ee = B[:3, 3]
        ee_axes = np.array([np.vstack([ee, ee + 0.10 * B[:3, i]]) for i in range(3)])

        # Prediction used by step 5: FK + hand-eye, not the PnP pose.
        C_pred = invert(T_camera_to_base) @ B @ T_board_to_ee
        proj = ideal.project(transform_points(C_pred, world_points))
        err = np.linalg.norm(proj - image_points[k], axis=1)
        pred_means.append(float(err.mean()))
        pred_maxs.append(float(err.max()))

        overlay = reports_dir / "step5_overlay" / images[image_idx].name
        image_path = overlay if overlay.exists() else images[image_idx]

        poses.append(
            {
                "view": k + 1,
                "image_name": images[image_idx].name,
                "image": _image_data_uri(image_path),
                "links": np.round(np.asarray(link_pts), 5).tolist(),
                "plane": {
                    "x": plane_pts[:, 0].tolist(),
                    "y": plane_pts[:, 1].tolist(),
                    "z": plane_pts[:, 2].tolist(),
                },
                "grid": _grid_xyz(board_pts.reshape(rows, cols, 3)),
                "corners": {
                    "x": plane_pts[:, 0].tolist(),
                    "y": plane_pts[:, 1].tolist(),
                    "z": plane_pts[:, 2].tolist(),
                },
                "ee_axes": {
                    "x": [[float(p[0]) for p in axis] for axis in ee_axes],
                    "y": [[float(p[1]) for p in axis] for axis in ee_axes],
                    "z": [[float(p[2]) for p in axis] for axis in ee_axes],
                },
                "centre": np.round(T_board_to_base[:3, 3], 5).tolist(),
                "pnp_rmse": float(pnp_rmse[k]),
                "pred_mean": float(err.mean()),
                "pred_max": float(err.max()),
                "joints": np.round(joints[image_idx], 3).tolist(),
            }
        )

    all_boards_x, all_boards_y, all_boards_z = _segments(outline_a, outline_b)
    first = poses[0]
    rpy = euler_zyx_deg(T_camera_to_base[:3, :3])
    quat = matrix_to_quat(T_camera_to_base[:3, :3])
    theta = np.linspace(0.0, 2.0 * np.pi, 81)
    ring = 0.42 * np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    payload = {
        "poses": poses,
        "trace": {
            "links": {
                "x": [p[0] for p in first["links"]],
                "y": [p[1] for p in first["links"]],
                "z": [p[2] for p in first["links"]],
            },
            "joints": {
                "x": [p[0] for p in first["links"]],
                "y": [p[1] for p in first["links"]],
                "z": [p[2] for p in first["links"]],
            },
            "plane": first["plane"],
            "grid": first["grid"],
            "corners": first["corners"],
            "ee_axes": first["ee_axes"],
        },
        "all": {
            "boards": {"x": all_boards_x, "y": all_boards_y, "z": all_boards_z},
            "centres": {
                "x": board_centres[:, 0].tolist(),
                "y": board_centres[:, 1].tolist(),
                "z": board_centres[:, 2].tolist(),
                "text": [f"Pose {k + 1} · PnP {pnp_rmse[k]:.2f} px" for k in range(n_views)],
            },
            "ee_path": {
                "x": ee_positions[:, 0].tolist(),
                "y": ee_positions[:, 1].tolist(),
                "z": ee_positions[:, 2].tolist(),
                "text": [f"Pose {k + 1} · EE" for k in range(n_views)],
            },
            "base_ring": {"x": ring[:, 0].tolist(), "y": ring[:, 1].tolist(), "z": ring[:, 2].tolist()},
        },
        "static": {
            "camera_pos": cam_origin.tolist(),
            "camera_R": T_camera_to_base[:3, :3].tolist(),
            "camera_matrix": T_camera_to_base.tolist(),
            "frustum": frustum_xyz,
            "body": camera_body,
            "depth": depth,
            "gt_frustum": gt_frustum,
            "gt_marker": gt_marker,
            "has_gt": gt_path.exists(),
        },
        "summary": {
            "method": str(res["method"]),
            "views": n_views,
            "translation": [round(float(v), 4) for v in cam_origin],
            "rpy": [round(float(v), 3) for v in rpy],
            "quat": [round(float(v), 5) for v in quat],
            "matrix": [[round(float(v), 5) for v in row] for row in T_camera_to_base],
            "pnp_mean": round(float(np.mean(pnp_rmse)), 3),
            "pnp_max": round(float(np.max(pnp_rmse)), 3),
            "pred_mean": round(float(np.mean(pred_means)), 3),
            "pred_max": round(float(np.max(pred_maxs)), 3),
            "gt_rot": None if gt_rot is None else round(float(np.degrees(gt_rot)), 3),
            "gt_trans": None if gt_trans is None else round(float(gt_trans * 1000.0), 3),
            "has_gt": gt_path.exists(),
        },
    }

    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = (
        _HTML_TEMPLATE.replace("__PLOTLY_SCRIPT__", _plotly_script(args.cdn))
        .replace("__DATA__", payload_json)
        .replace("__DEPTH__", f"{depth:.2f}")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"views            : {n_views}")
    print(f"camera position  : [{cam_origin[0]:.4f}, {cam_origin[1]:.4f}, {cam_origin[2]:.4f}] m")
    print(f"camera RPY       : [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}] deg")
    print(f"PnP RMSE         : {np.mean(pnp_rmse):.2f} px mean / {np.max(pnp_rmse):.2f} px max")
    print(f"prediction RMSE  : {np.mean(pred_means):.2f} px mean / {np.max(pred_maxs):.2f} px max")
    print("drag = orbit, scroll = zoom, right-drag = pan, arrows = next/previous pose")
    if args.open:
        import webbrowser

        webbrowser.open(out_path.resolve().as_uri())


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visualisasi Kalibrasi Eye-to-Hand</title>
__PLOTLY_SCRIPT__
<style>
  :root {
    --bg:#070b14; --panel:#0b1220; --panel-2:#0f1a2e; --line:#1e293b;
    --ink:#e5edf8; --muted:#93a4bb; --accent:#38bdf8; --accent-2:#a78bfa;
    --good:#34d399; --rose:#fb7185; --amber:#fbbf24;
    --mono:"Cascadia Code","SFMono-Regular",Consolas,"Liberation Mono",monospace;
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body {
    margin:0; color:var(--ink); overflow:hidden;
    font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif;
    background:radial-gradient(1100px 600px at 18% -10%, #12213a 0%, var(--bg) 55%);
    display:flex; flex-direction:column;
  }
  header {
    display:flex; align-items:flex-start; justify-content:space-between; gap:18px;
    padding:14px 22px 13px; border-bottom:1px solid var(--line);
    background:rgba(9,15,27,.82); backdrop-filter:blur(8px);
  }
  h1 { margin:0; font-size:19px; font-weight:650; letter-spacing:-.25px; }
  h1 span { color:var(--accent); }
  .sub { margin-top:5px; color:var(--muted); font-size:12.5px; }
  .badges { display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; max-width:1000px; }
  .badge {
    border:1px solid #21314b; background:#0d1930; color:#c7d7ec;
    border-radius:999px; padding:4px 10px; font-size:11.5px; white-space:nowrap;
    font-variant-numeric:tabular-nums;
  }
  .badge.good { border-color:#155e4b; background:#08251d; color:#6ee7b7; }
  .badge.hot { border-color:#5b2637; background:#26101a; color:#fda4af; }
  .toolbar {
    display:flex; align-items:center; gap:8px; flex-wrap:wrap;
    padding:9px 16px; border-bottom:1px solid var(--line); background:var(--panel);
  }
  .group { display:flex; align-items:center; gap:7px; }
  button {
    border:1px solid #26364f; background:#0d1729; color:#dbe7f6;
    border-radius:8px; padding:6px 11px; font-size:13px; cursor:pointer;
    transition:background .15s, border-color .15s;
  }
  button:hover { background:#14233b; border-color:#3b557c; }
  button.primary { border-color:#0e7490; background:#082f49; color:#a5f3fc; }
  button.primary.active { border-color:#b45309; background:#3b210a; color:#fcd34d; }
  input[type=range] {
    width:min(30vw, 360px); min-width:150px; accent-color:var(--accent);
    background:transparent;
  }
  .spacer { flex:1; }
  .toolbar label { display:flex; align-items:center; gap:5px; font-size:12px; color:var(--muted); cursor:pointer; }
  .toolbar input[type=checkbox] { accent-color:var(--accent); }
  #viewLabel { color:var(--muted); font-size:12px; min-width:86px; text-align:center; font-variant-numeric:tabular-nums; }
  .wrap { flex:1; min-height:0; display:grid; grid-template-columns:minmax(0,1fr) 372px; }
  #scene { min-width:0; min-height:0; height:100%; }
  #panel {
    min-height:0; overflow-y:auto; padding:15px; border-left:1px solid var(--line);
    background:linear-gradient(180deg, #0b1424 0%, #09101d 100%);
  }
  .pose-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
  .pose-head h2 { margin:0; font-size:15px; font-weight:600; }
  .chip { font-family:var(--mono); font-size:11px; color:#7dd3fc; background:#082f49; border:1px solid #0e7490; border-radius:6px; padding:3px 7px; }
  .imgwrap { background:#020617; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  #calImage { display:block; width:100%; height:auto; }
  .metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:11px; }
  .metric { border:1px solid var(--line); background:#0a1424; border-radius:9px; padding:8px 10px; }
  .metric span { display:block; color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.055em; }
  .metric b { display:block; margin-top:3px; font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }
  .metric b.good { color:var(--good); }
  .metric b.warn { color:var(--amber); }
  h3 { margin:16px 0 7px; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  .joints { display:grid; grid-template-columns:repeat(4, 1fr); gap:6px; }
  .joint { border:1px solid var(--line); border-radius:7px; background:#0a1424; padding:5px 6px; text-align:center; }
  .joint em { display:block; color:#64748b; font-style:normal; font-size:10px; }
  .joint b { display:block; font-size:12.5px; font-variant-numeric:tabular-nums; font-weight:500; }
  details { margin-top:13px; border:1px solid var(--line); border-radius:9px; background:#0a1424; }
  summary { cursor:pointer; padding:8px 10px; color:#b9c9dd; font-size:12px; user-select:none; }
  details[open] summary { border-bottom:1px solid var(--line); }
  details pre { margin:0; padding:9px 10px; font-family:var(--mono); font-size:10.5px; line-height:1.55; color:#a5f3fc; overflow-x:auto; }
  .quat { padding:0 10px 9px; font-family:var(--mono); font-size:10.5px; color:#94a3b8; }
  .hint { color:var(--muted); font-size:11.5px; line-height:1.55; margin:8px 0 0; }
  .legend { display:flex; flex-direction:column; gap:4px; color:var(--muted); font-size:11px; }
  .legend i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
  .error { padding:32px; color:#fda4af; font-size:14px; line-height:1.6; }
  @media (max-width: 980px) {
    body { overflow:auto; height:auto; }
    .wrap { display:block; }
    #scene { height:64vh; }
    #panel { border-left:0; border-top:1px solid var(--line); overflow:visible; }
    .badges { max-width:100%; justify-content:flex-start; }
    header { flex-direction:column; }
    input[type=range] { width:100%; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Visualisasi <span>Kalibrasi Eye-to-Hand</span></h1>
    <div class="sub">Kinova Gen3 · kamera tetap · checkerboard di end-effector · <span id="viewCount">–</span> pose · kedalaman frustum __DEPTH__ m</div>
  </div>
  <div class="badges" id="badges"></div>
</header>

<div class="toolbar">
  <div class="group">
    <button id="prev" title="Pose sebelumnya (←)">◀</button>
    <input id="slider" type="range" min="0" max="0" value="0" aria-label="Pilih pose">
    <button id="next" title="Pose berikutnya (→)">▶</button>
    <button id="play" class="primary" title="Animasi semua pose">▶ Jalankan</button>
  </div>
  <span id="viewLabel">–</span>
  <span class="spacer"></span>
  <label><input id="showAll" type="checkbox" checked> semua pose</label>
  <label><input id="showFrustum" type="checkbox" checked> frustum kamera</label>
  <label><input id="showTrail" type="checkbox" checked> lintasan EE</label>
  <label id="gtToggle"><input id="showGT" type="checkbox"> ground truth</label>
</div>

<div class="wrap">
  <div id="scene"></div>
  <aside id="panel">
    <div class="pose-head">
      <h2 id="poseTitle">Pose 1</h2>
      <span class="chip" id="imageName">–</span>
    </div>
    <div class="imgwrap"><img id="calImage" alt="gambar kalibrasi"></div>

    <div class="metrics">
      <div class="metric"><span>PnP RMSE</span><b id="pnp">–</b></div>
      <div class="metric"><span>prediksi RMSE</span><b id="pred" class="good">–</b></div>
      <div class="metric"><span>prediksi maks</span><b id="predmax">–</b></div>
      <div class="metric"><span>kamera → papan</span><b id="distance">–</b></div>
    </div>

    <details>
      <summary>Estimasi T<sub>camera→base</sub></summary>
      <pre id="matrix"></pre>
      <div class="quat" id="quat"></div>
    </details>

    <h3>Sudut joint [deg]</h3>
    <div class="joints" id="joints"></div>

    <h3>Keterangan</h3>
    <div class="legend">
      <div><i style="background:#94a3b8"></i>robot (FK, URDF Kinova Gen3)</div>
      <div><i style="background:#38bdf8"></i>papan checkerboard di end-effector</div>
      <div><i style="background:#fb7185"></i>kamera terestimasi + frustum</div>
      <div><i style="background:#64748b"></i>pose lain / ground truth</div>
    </div>
    <p class="hint">
      Gambar: <b style="color:#6ee7b7">lingkaran hijau</b> = sudut terdeteksi,
      <b style="color:#fda4af">silang merah</b> = prediksi dari FK + hand-eye.
      Drag untuk orbit, scroll untuk zoom, klik marker pose untuk melompat.
    </p>
  </aside>
</div>

<script>
(function () {
  "use strict";
  const DATA = __DATA__;
  const POSES = DATA.poses;
  const S = DATA.summary;
  const T = DATA.trace;
  const A = DATA.all;
  const ST = DATA.static;
  const div = document.getElementById('scene');

  if (typeof Plotly === 'undefined') {
    div.innerHTML = '<div class="error"><b>Plotly gagal dimuat.</b><br>Gunakan koneksi internet atau jalankan ulang skrip tanpa <code>--cdn</code> agar Plotly disematkan ke file HTML.</div>';
    return;
  }

  const TRACE = {
    links: 0, joints: 1, plane: 2, grid: 3, corners: 4,
    eeX: 5, eeY: 6, eeZ: 7,
    frustum: 8, body: 9, camMarker: 10,
    camX: 11, camY: 12, camZ: 13,
    baseX: 14, baseY: 15, baseZ: 16,
    allBoards: 17, centres: 18, eePath: 19,
    gtFrustum: 20, gtMarker: 21, baseRing: 22
  };

  const COL = {
    robot: '#94a3b8', joint: '#e2e8f0',
    board: '#0ea5e9', boardGrid: '#7dd3fc', boardCorner: '#fbbf24',
    ee: ['#f87171', '#4ade80', '#60a5fa'],
    camera: '#fb7185', camAxis: ['#fda4af', '#fde68a', '#67e8f9'],
    baseAxis: ['#ef4444', '#22c55e', '#3b82f6'],
    ghost: '#475569', ghostPoint: '#64748b', trail: '#a78bfa',
    gt: '#94a3b8'
  };

  function lineTrace(x, y, z, color, width, dash, opacity) {
    return {
      type: 'scatter3d', mode: 'lines', x: x, y: y, z: z,
      line: { color: color, width: width, dash: dash || 'solid' },
      opacity: opacity === undefined ? 1 : opacity,
      hoverinfo: 'skip', showlegend: false
    };
  }

  function markerTrace(x, y, z, color, size, symbol, text, textpos) {
    const trace = {
      type: 'scatter3d', mode: text ? 'markers+text' : 'markers',
      x: x, y: y, z: z,
      marker: { color: color, size: size, symbol: symbol || 'circle', line: { color: color, width: 1 } },
      showlegend: false, hoverinfo: 'skip'
    };
    if (text) {
      trace.text = text;
      trace.textposition = textpos || 'top center';
      trace.textfont = { size: 10, color: color };
    }
    return trace;
  }

  function hoverTrace(x, y, z, color, text) {
    return {
      type: 'scatter3d', mode: 'markers', x: x, y: y, z: z,
      text: text, hoverinfo: 'text', hoverlabel: { bgcolor: '#0f1a2e', bordercolor: '#334155', font: { color: '#dbeafe', size: 11 } },
      marker: { color: color, size: 5, symbol: 'x', opacity: 0.9 },
      showlegend: false
    };
  }

  function axisTraces(origin, R, len, colors, width) {
    return [0, 1, 2].map(function (i) {
      const p = [origin[0] + len * R[0][i], origin[1] + len * R[1][i], origin[2] + len * R[2][i]];
      return lineTrace([origin[0], p[0]], [origin[1], p[1]], [origin[2], p[2]], colors[i], width);
    });
  }

  function meshTrace(m, color, opacity) {
    return {
      type: 'mesh3d', x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k,
      color: color, opacity: opacity, flatshading: true,
      hoverinfo: 'skip', showlegend: false
    };
  }

  const traces = [
    lineTrace(T.links.x, T.links.y, T.links.z, COL.robot, 8),
    markerTrace(T.joints.x, T.joints.y, T.joints.z, COL.joint, 4, 'circle'),
    meshTrace(T.plane, COL.board, 0.30),
    lineTrace(T.grid.x, T.grid.y, T.grid.z, COL.boardGrid, 2.5),
    markerTrace(T.corners.x, T.corners.y, T.corners.z, COL.boardCorner, 6, 'diamond'),
    lineTrace(T.ee_axes.x[0], T.ee_axes.y[0], T.ee_axes.z[0], COL.ee[0], 5),
    lineTrace(T.ee_axes.x[1], T.ee_axes.y[1], T.ee_axes.z[1], COL.ee[1], 5),
    lineTrace(T.ee_axes.x[2], T.ee_axes.y[2], T.ee_axes.z[2], COL.ee[2], 5),
    lineTrace(ST.frustum.x, ST.frustum.y, ST.frustum.z, COL.camera, 2.5, 'dot'),
    meshTrace(ST.body, COL.camera, 0.30),
    markerTrace([ST.camera_pos[0]], [ST.camera_pos[1]], [ST.camera_pos[2]], COL.camera, 9, 'diamond', ['kamera'], 'middle right'),
    ...axisTraces(ST.camera_pos, ST.camera_R, 0.16, COL.camAxis, 4),
    ...axisTraces([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.22, COL.baseAxis, 5),
    lineTrace(A.boards.x, A.boards.y, A.boards.z, COL.ghost, 1.0, 'solid', 0.55),
    hoverTrace(A.centres.x, A.centres.y, A.centres.z, COL.ghostPoint, A.centres.text),
    {
      type: 'scatter3d', mode: 'lines+markers',
      x: A.ee_path.x, y: A.ee_path.y, z: A.ee_path.z,
      line: { color: COL.trail, width: 3 }, marker: { color: COL.trail, size: 3.5 },
      text: A.ee_path.text, hoverinfo: 'text', showlegend: false
    },
    lineTrace(ST.gt_frustum.x, ST.gt_frustum.y, ST.gt_frustum.z, COL.gt, 1.6, 'dash', 0.75),
    markerTrace(ST.gt_marker.x, ST.gt_marker.y, ST.gt_marker.z, COL.gt, 8, 'diamond-open'),
    lineTrace(A.base_ring.x, A.base_ring.y, A.base_ring.z, '#1f2f49', 2)
  ];

  // The two ground-truth traces start hidden; the badge still reports the error.
  traces[TRACE.gtFrustum].visible = false;
  traces[TRACE.gtMarker].visible = false;

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: '#070b14',
    plot_bgcolor: '#070b14',
    showlegend: false,
    scene: {
      bgcolor: '#070b14',
      aspectmode: 'data',
      dragmode: 'orbit',
      xaxis: { title: { text: 'X [m]', font: { size: 11, color: '#93a4bb' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#16233a', zerolinecolor: '#27364f', showbackground: true, backgroundcolor: '#080d18' },
      yaxis: { title: { text: 'Y [m]', font: { size: 11, color: '#93a4bb' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#16233a', zerolinecolor: '#27364f', showbackground: true, backgroundcolor: '#080d18' },
      zaxis: { title: { text: 'Z [m]', font: { size: 11, color: '#93a4bb' } }, tickfont: { size: 10, color: '#64748b' }, gridcolor: '#16233a', zerolinecolor: '#27364f', showbackground: true, backgroundcolor: '#080d18' },
      camera: { eye: { x: 1.30, y: -1.51, z: 0.66 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0.02 } }
    }
  };

  const config = { responsive: true, displaylogo: false, scrollZoom: true };

  document.getElementById('viewCount').textContent = S.views;
  document.getElementById('slider').max = POSES.length - 1;
  document.getElementById('viewLabel').textContent = 'pose 1 / ' + POSES.length;
  document.getElementById('gtToggle').style.display = ST.has_gt ? 'flex' : 'none';

  const fmt = function (v, n) { return Number(v).toFixed(n); };
  const badge = function (text, cls) { return '<span class="badge ' + (cls || '') + '">' + text + '</span>'; };
  document.getElementById('badges').innerHTML = [
    badge('metode: ' + S.method),
    badge('kamera: [' + S.translation.map(function (v) { return fmt(v, 3); }).join(', ') + '] m', 'good'),
    badge('RPY: [' + S.rpy.map(function (v) { return fmt(v, 2); }).join(', ') + ']°'),
    badge('PnP: ' + fmt(S.pnp_mean, 2) + ' px'),
    badge('validasi: ' + fmt(S.pred_mean, 2) + ' px'),
    S.has_gt ? badge('GT: ' + fmt(S.gt_rot, 2) + '° / ' + fmt(S.gt_trans, 2) + ' mm', 'good') : ''
  ].join('');

  document.getElementById('matrix').textContent = S.matrix.map(function (row) {
    return row.map(function (v) { return fmt(v, 5).padStart(10); }).join(' ');
  }).join('\n');
  document.getElementById('quat').textContent = 'quat (w,x,y,z) = [' + S.quat.map(function (v) { return fmt(v, 5); }).join(', ') + ']';

  let current = 0;
  let timer = null;
  const playBtn = document.getElementById('play');

  function stopPlayback() {
    if (timer) {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = '▶ Jalankan';
      playBtn.classList.remove('active');
    }
  }

  function refreshPanel(i) {
    const p = POSES[i];
    document.getElementById('poseTitle').textContent = 'Pose ' + p.view;
    document.getElementById('imageName').textContent = p.image_name;
    document.getElementById('calImage').src = p.image;
    document.getElementById('pnp').textContent = fmt(p.pnp_rmse, 2) + ' px';
    document.getElementById('pred').textContent = fmt(p.pred_mean, 2) + ' px';
    document.getElementById('predmax').textContent = fmt(p.pred_max, 2) + ' px';
    const d = Math.hypot(p.centre[0] - ST.camera_pos[0], p.centre[1] - ST.camera_pos[1], p.centre[2] - ST.camera_pos[2]);
    document.getElementById('distance').textContent = fmt(d, 3) + ' m';
    const predEl = document.getElementById('pred');
    predEl.className = p.pred_mean < 1 ? 'good' : (p.pred_mean < 3 ? '' : 'warn');
    document.getElementById('joints').innerHTML = p.joints.map(function (v, j) {
      return '<span class="joint"><em>q' + (j + 1) + '</em><b>' + fmt(v, 2) + '</b></span>';
    }).join('');
    document.getElementById('slider').value = String(i);
    document.getElementById('viewLabel').textContent = 'pose ' + (i + 1) + ' / ' + POSES.length;
  }

  function setPose(i, fromPlayback) {
    if (!fromPlayback) stopPlayback();
    current = ((i % POSES.length) + POSES.length) % POSES.length;
    const p = POSES[current];
    const xs = p.links.map(function (q) { return q[0]; });
    const ys = p.links.map(function (q) { return q[1]; });
    const zs = p.links.map(function (q) { return q[2]; });

    Plotly.restyle(div, { x: [xs, xs], y: [ys, ys], z: [zs, zs] }, [TRACE.links, TRACE.joints]);
    Plotly.restyle(div, { x: [p.plane.x], y: [p.plane.y], z: [p.plane.z] }, [TRACE.plane]);
    Plotly.restyle(div, { x: [p.grid.x], y: [p.grid.y], z: [p.grid.z] }, [TRACE.grid]);
    Plotly.restyle(div, { x: [p.corners.x], y: [p.corners.y], z: [p.corners.z] }, [TRACE.corners]);
    Plotly.restyle(div, { x: p.ee_axes.x, y: p.ee_axes.y, z: p.ee_axes.z }, [TRACE.eeX, TRACE.eeY, TRACE.eeZ]);

    refreshPanel(current);
  }

  function visible(indices, on) {
    Plotly.restyle(div, { visible: on }, indices);
  }

  Plotly.newPlot(div, traces, layout, config).then(function () {
    refreshPanel(0);
  });

  div.on('plotly_click', function (event) {
    const pt = event.points && event.points[0];
    if (!pt) return;
    if (pt.curveNumber === TRACE.centres || pt.curveNumber === TRACE.eePath) {
      setPose(pt.pointNumber);
    }
  });

  document.getElementById('slider').addEventListener('input', function (e) {
    stopPlayback();
    setPose(Number(e.target.value));
  });
  document.getElementById('prev').addEventListener('click', function () { setPose(current - 1); });
  document.getElementById('next').addEventListener('click', function () { setPose(current + 1); });
  playBtn.addEventListener('click', function () {
    if (timer) { stopPlayback(); return; }
    playBtn.textContent = '⏸ Jeda';
    playBtn.classList.add('active');
    timer = setInterval(function () { setPose(current + 1, true); }, 1250);
  });
  document.getElementById('showAll').addEventListener('change', function (e) {
    visible([TRACE.allBoards, TRACE.centres], e.target.checked);
  });
  document.getElementById('showFrustum').addEventListener('change', function (e) {
    visible([TRACE.frustum], e.target.checked);
  });
  document.getElementById('showTrail').addEventListener('change', function (e) {
    visible([TRACE.eePath], e.target.checked);
  });
  document.getElementById('showGT').addEventListener('change', function (e) {
    visible([TRACE.gtFrustum, TRACE.gtMarker], e.target.checked);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { e.preventDefault(); setPose(current - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); setPose(current + 1); }
    if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

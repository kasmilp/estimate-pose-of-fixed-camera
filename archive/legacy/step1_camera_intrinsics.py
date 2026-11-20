"""Step 1 - Camera intrinsics (MATLAB: Camera Calibrator app).

Priority order
--------------
1. ``--intrinsics path/to/intrinsics.npz`` (K, distortion, image_size)
2. ``<data-dir>/intrinsics.npz`` if it exists (produced by generate_data.py)
3. Calibrate from chessboard images in ``<data-dir>/intrinsics_images``
4. ``intrinsics_estimation=True`` -> use the classic 9x6 chessboard images as
   calibration images (single-camera calibration, OpenCV ``calibrateCamera``)
5. ``--fallback`` -> rough pinhole guess (fx = fy = 1.2 * width, principal
   point at the image center). Only for smoke tests, never for real work.

Saves ``<data-dir>/intrinsics.npz`` and ``reports/intrinsics_summary.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from posekit.camera import CameraIntrinsics, Checkerboard, detect_checkerboard
from posekit.io import list_images


def calibrate_from_images(image_paths, board: Checkerboard, image_size=None):
    object_points, image_points = [], []
    used = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        corners = detect_checkerboard(image, board)
        if corners is None:
            print(f"[skip] no checkerboard: {Path(path).name}")
            continue
        object_points.append(board.world_points())
        image_points.append(corners)
        used.append(Path(path).name)
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])

    if len(object_points) < 4:
        raise RuntimeError(f"need >= 4 valid views, got {len(object_points)}")

    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    print(f"[ok] calibrated on {len(used)} images, RMS reprojection error = {rms:.3f} px")
    return CameraIntrinsics(K=K, distortion=dist.ravel(), image_size=image_size), used, rms


def fallback_intrinsics(image_size=(1280, 720)) -> CameraIntrinsics:
    width, height = image_size
    f = 1.2 * width
    return CameraIntrinsics(
        K=np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float),
        distortion=np.zeros(5),
        image_size=image_size,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration", help="folder with the calibration images")
    parser.add_argument("--intrinsics", default=None, help="explicit intrinsics .npz file")
    parser.add_argument("--board", default="9x6", help="inner corners of the chessboard, e.g. 9x6")
    parser.add_argument("--square-size", type=float, default=0.02, help="square size in meters")
    parser.add_argument("--use-calibration-images", action="store_true", help="estimate intrinsics from the calibration images themselves")
    parser.add_argument("--fallback", action="store_true", help="write a rough pinhole guess if nothing else is available")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    cols, rows = (int(v) for v in args.board.lower().split("x"))
    board = Checkerboard(pattern_size=(cols, rows), square_size=args.square_size)
    out_path = data_dir / "intrinsics.npz"
    report_path = Path(args.reports_dir) / "intrinsics_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    source, used, rms = "", [], None
    if args.intrinsics:
        source = f"provided file {args.intrinsics}"
        intrinsics = CameraIntrinsics.from_calibration_file(args.intrinsics)
    elif out_path.exists():
        source = f"existing {out_path}"
        intrinsics = CameraIntrinsics.from_calibration_file(out_path)
    else:
        candidates = list_images(data_dir / "intrinsics_images")
        if not candidates and args.use_calibration_images:
            candidates = list_images(data_dir / "images")
        if candidates:
            source = f"single-camera calibration on {len(candidates)} images"
            intrinsics, used, rms = calibrate_from_images(candidates, board)
        elif args.fallback:
            source = "FALLBACK pinhole guess (smoke test only)"
            intrinsics = fallback_intrinsics()
        else:
            raise SystemExit(
                "No intrinsics available.\n"
                "  - put intrinsics.npz in the data dir, or\n"
                "  - provide chessboard images in <data-dir>/intrinsics_images, or\n"
                "  - rerun with --fallback (smoke test only)."
            )

    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=intrinsics.K,
        distortion=intrinsics.distortion,
        image_size=np.array(intrinsics.image_size),
        board_pattern_size=np.array(board.pattern_size),
        board_square_size=board.square_size,
    )

    lines = [
        "Camera intrinsics",
        "=================",
        f"source            : {source}",
        f"images used       : {', '.join(used) if used else '-'}",
        f"calibration RMS   : {f'{rms:.3f} px' if rms is not None else '-'}",
        f"image size        : {intrinsics.image_size}",
        f"fx, fy            : {intrinsics.fx:.2f}, {intrinsics.fy:.2f}",
        f"cx, cy            : {intrinsics.cx:.2f}, {intrinsics.cy:.2f}",
        f"distortion        : {np.round(intrinsics.distortion, 5).tolist()}",
        f"board (inner)     : {board.pattern_size[0]}x{board.pattern_size[1]}",
        f"square size [m]   : {board.square_size}",
        f"saved to          : {out_path}",
    ]
    text = "\n".join(lines)
    print(text)
    report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Step 2 - Collect hand-eye calibration data.

What you need for an *eye-to-hand* setup (camera fixed, board on the robot):

    1. Mount the calibration board on the end effector.
    2. Move the robot to a pose where the board is fully visible.
    3. Capture the image and record the joint configuration.
    4. Repeat >= 15 times, covering the workspace (different orientations!).

Folder layout expected by the rest of the pipeline:

    data/calibration/
        images/1.jpg, 2.jpg, ...            # image k <-> row k of joint_angles.txt
        joint_angles.txt                    # N x 7, degrees
        intrinsics.npz                      # step1 output

The joint angles are the **raw encoders** (0..360 deg, not wrapped to
[-180, 180]) exactly like ``jointAngles.mat`` in the MathWorks example.

This script can:
    * validate an existing dataset,
    * import ``calibrationData.zip`` / ``jointAngles.mat`` from MathWorks,
    * generate a synthetic dataset so the pipeline can be tested offline.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from posekit.io import list_images
from posekit.robot import RobotModel
from posekit.se3 import rotation_angle


def load_joint_angles(path: Path) -> np.ndarray:
    if path.suffix == ".mat":
        from scipy.io import loadmat

        mat = loadmat(path)
        key = next(k for k in mat if not k.startswith("__"))
        return np.atleast_2d(np.asarray(mat[key], dtype=float))
    return np.atleast_2d(np.loadtxt(path, delimiter=None))


def validate(data_dir: Path, show_rotation_coverage: bool = True) -> bool:
    images = list_images(data_dir / "images")
    joints_path = data_dir / "joint_angles.txt"
    ok = True

    print(f"images            : {len(images)}")
    if not joints_path.exists():
        print("[error] joint_angles.txt is missing")
        return False
    angles = load_joint_angles(joints_path)
    print(f"joint configurations: {angles.shape[0]} x {angles.shape[1]}")

    if len(images) != angles.shape[0]:
        print(f"[error] {len(images)} images vs {angles.shape[0]} joint rows -> they must match 1:1")
        ok = False
    if angles.shape[1] != 7:
        print("[warn] expected 7 joints for the Kinova Gen3")
    if angles.shape[0] < 15:
        print("[warn] fewer than 15 poses; the MathWorks example recommends >= 15")

    if not (data_dir / "intrinsics.npz").exists():
        print("[warn] intrinsics.npz not found - run step1_camera_intrinsics.py")

    if show_rotation_coverage:
        model = RobotModel()
        poses = model.fk_many(angles)
        angles_deg = [np.degrees(rotation_angle(poses[i][:3, :3] @ poses[j][:3, :3].T)) for i in range(len(poses)) for j in range(i + 1, len(poses))]
        angles_deg = np.array(angles_deg)
        print(f"pairwise EE rotation: min {angles_deg.min():.1f} deg, median {np.median(angles_deg):.1f} deg, max {angles_deg.max():.1f} deg")
        small = int((angles_deg < 10).sum())
        print(f"pairs with < 10 deg rotation: {small} (a few of these are harmless, too many weakens the solve)")
    return ok


def import_mathworks_zip(zip_path: Path, data_dir: Path) -> None:
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        mat_name = next((n for n in names if n.lower().endswith(".mat")), None)
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                target = images_dir / Path(name).name
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        if mat_name is None:
            raise SystemExit("no jointAngles .mat inside the zip")
        with zf.open(mat_name) as src:
            tmp = data_dir / Path(mat_name).name
            tmp.write_bytes(src.read())
    angles = load_joint_angles(tmp)
    np.savetxt(data_dir / "joint_angles.txt", angles, fmt="%.4f")
    tmp.unlink()
    print(f"[ok] imported {len(list_images(images_dir))} images and {angles.shape[0]} joint rows into {data_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration")
    parser.add_argument("--from-zip", default=None, help="import calibrationData.zip downloaded from MathWorks")
    parser.add_argument("--generate-synthetic", action="store_true", help="render a synthetic dataset for testing")
    parser.add_argument("--noise-px", type=float, default=0.3, help="corner noise for the synthetic renderer")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.from_zip:
        import_mathworks_zip(Path(args.from_zip), data_dir)
    if args.generate_synthetic:
        from posekit.synthetic import generate

        generate(data_dir, noise_px=args.noise_px)
        print(f"[ok] synthetic dataset written to {data_dir}")

    if not (data_dir / "images").exists():
        raise SystemExit(f"no images in {data_dir / 'images'}")
    validate(data_dir)


if __name__ == "__main__":
    main()

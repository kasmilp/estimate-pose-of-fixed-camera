"""Export the calibration dataset as plain text / .mat for the MATLAB example.

The MathWorks example expects:

    calibrationData/
        *.jpg              # calibration images
        jointAngles.mat    # N x 7, degrees, row k <-> image k

and, for the intrinsics, ``estimateCameraPoseWrtRobotBase.mat`` with
``intrinsics`` (cameraIntrinsics object) and ``checkerboardSquareSizeInMeters``.

This script writes:

    <out>/calibrationData.zip        # images + jointAngles.mat
    <out>/estimateCameraPoseWrtRobotBase.mat   # intrinsics + square size

so the Python results can be compared 1:1 with MATLAB.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from posekit.io import list_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/calibration_synth")
    parser.add_argument("--out-dir", default="exports")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scipy.io import savemat

    joint_angles = np.loadtxt(data_dir / "joint_angles.txt", ndmin=2)
    meta = np.load(data_dir / "intrinsics.npz")

    # --- calibrationData.zip -------------------------------------------------
    staging = out_dir / "calibrationData"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for image in list_images(data_dir / "images"):
        shutil.copy2(image, staging / image.name)
    savemat(staging / "jointAngles.mat", {"jointAngles": joint_angles})

    zip_path = out_dir / "calibrationData.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in staging.iterdir():
            zf.write(file, f"calibrationData/{file.name}")

    # --- estimateCameraPoseWrtRobotBase.mat ----------------------------------
    K = meta["K"]
    dist = meta["distortion"].ravel()
    intr = {
        "FocalLength": np.array([K[0, 0], K[1, 1]]),
        "PrincipalPoint": np.array([K[0, 2], K[1, 2]]),
        "RadialDistortion": dist[:2] if dist.size >= 2 else np.zeros(2),
        "TangentialDistortion": dist[2:4] if dist.size >= 4 else np.zeros(2),
        "ImageSize": np.array(meta["image_size"], dtype=float),
        "ReprojectionErrors": np.zeros((0, 2)),
    }
    mat_path = out_dir / "estimateCameraPoseWrtRobotBase.mat"
    savemat(
        mat_path,
        {
            "intrinsics": intr,
            "checkerboardSquareSizeInMeters": float(meta["board_square_size"]),
            "checkerboardPatternSize": np.array(meta["board_pattern_size"]),
        },
    )
    print(f"wrote {zip_path} ({joint_angles.shape[0]} images)")
    print(f"wrote {mat_path}")
    print(
        "\nMATLAB side:\n"
        f"  unzip('{zip_path.name}')\n"
        f"  load('{mat_path.name}','intrinsics','checkerboardSquareSizeInMeters')\n"
        "  load('calibrationData/jointAngles')\n"
        "  % ... then run the example script"
    )


if __name__ == "__main__":
    main()

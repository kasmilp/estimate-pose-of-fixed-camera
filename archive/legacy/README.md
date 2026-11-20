# Estimate Pose of a Fixed Camera Relative to the Robot Base (Python port)

Python re-implementation of the MathWorks Robotics System Toolbox example
[**Estimate Pose of Fixed Camera Relative to Robot Base**](https://www.mathworks.com/help/robotics/ug/estimate-pose-of-fixed-camera-relative-to-robot-base.html)
(R2024b+), i.e. **eye-to-hand** (stationary camera) hand-eye calibration of a
Kinova Gen3 with an OpenCV/scipy stack.

> MATLAB needs Robotics System Toolbox + Computer Vision Toolbox and the
> `calibrationData.zip` dataset. This repo works with the same dataset **or**
> with a synthetic dataset generated offline, so every step can be run and
> verified end-to-end.

---

## 1. The problem

A camera is mounted on a fixed beam. To pick objects, a detection in the
camera frame must be expressed in the robot base frame:

```
p_base = T_camera_to_base @ p_camera
```

`T_camera_to_base` is what we estimate. It cannot be measured with a ruler
(the optical center / image-sensor plane is not marked on the camera body),
so it is solved with hand-eye calibration:

* the camera is **fixed** (eye-to-hand, `"stationary-camera"`),
* a checkerboard is mounted on the **end effector**,
* for each robot pose `i` we know
  * `T_board_to_camera_i`  – camera extrinsics (`estimateExtrinsics` / `solvePnP`),
  * `T_end_effector_to_base_i` – forward kinematics (`getTransform`).

For every pair of poses the *rigid-motion* constraint holds:

```
(B_j B_i^-1) · X = X · (C_j C_i^-1)          (the AX = XB equation)

B = T_end_effector_to_base ,  C = T_board_to_camera ,  X = T_camera_to_base
```

The pair direction matters: use `j` after `i` on **both** sides. Mixing
`B_j B_i^-1` with `C_i^-1 C_j` silently solves the eye-in-hand problem and
returns a ~180° wrong camera pose.

---

## 2. Pipeline (progress checklist)

| # | MATLAB example step | Python file | Status |
|---|---------------------|-------------|--------|
| 1 | Estimate camera intrinsics (`Camera Calibrator` app) | `step1_camera_intrinsics.py` | ✅ |
| 2 | Collect hand-eye data (`calibrationData.zip`, `jointAngles.mat`) | `step2_collect_data.py` | ✅ |
| 3 | Camera extrinsics + end-effector poses (`detectCheckerboardPoints`, `estimateExtrinsics`, `getTransform`) | `step3_extrinsics_and_fk.py` | ✅ |
| 4 | Solve the problem (`estimateCameraRobotTransform(...,"stationary-camera")`) + nonlinear refinement | `step4_solve_handeye.py` | ✅ |
| 5 | Validate / visualize (`TWorldToBase`, `plotCamera`, `show`) | `step5_validate_and_visualize.py` | ✅ |
| – | Offline sanity check with known ground truth | `selftest_synthetic.py` | ✅ |

### MATLAB → Python mapping

| MATLAB | Python (this repo) |
|--------|--------------------|
| `loadrobot("kinovaGen3")`, `show(robot,q)` | `posekit.robot.RobotModel` (URDF `data/kinova_gen3_7dof.urdf`) |
| `getTransform(kinova,q,"EndEffector_Link")` | `RobotModel.fk(q)` / `posekit.robot.get_transform` |
| `undistortImage` | `cv2.undistort` (`posekit.camera.undistort_image`) |
| `detectCheckerboardPoints(img, PartialDetections=false)` | `posekit.camera.detect_checkerboard` (`findChessboardCorners` + `cornerSubPix`) |
| `patternWorldPoints("checkerboard",sz,squareSize)` | `Checkerboard.world_points()` |
| `estimateExtrinsics(imgpts,worldpts,intrinsics)` | `posekit.camera.estimate_extrinsics` (`solvePnP` IPPE + LM refinement) |
| `estimateCameraRobotTransform(C,B,"stationary-camera")` | `posekit.handeye.estimate_camera_robot_transform` |
| `rigidtform3d(A*C)` / `transformPointsForward` | `T = Tc2b @ C`, `posekit.se3.transform_points` |
| `plotCamera`, `show`, `scatter3` | `posekit.plotting` (matplotlib) |

---

## 3. Quick start

```powershell
python -m pip install -r requirements.txt

# A) real dataset from MathWorks (recommended for the "official" numbers)
#    download calibrationData.zip + estimateCameraPoseWrtRobotBase.mat
python step2_collect_data.py --data-dir data/calibration --from-zip path\to\calibrationData.zip
python step1_camera_intrinsics.py --data-dir data/calibration        # or --fallback for a smoke test

# B) synthetic dataset (no MathWorks download needed, ground truth included)
python step2_collect_data.py --data-dir data/calibration_synth --generate-synthetic

# 2) run the pipeline
python step1_camera_intrinsics.py   --data-dir data/calibration_synth
python step3_extrinsics_and_fk.py   --data-dir data/calibration_synth
python step4_solve_handeye.py       --data-dir data/calibration_synth   # park_numpy + nonlinear refinement
python step5_validate_and_visualize.py --data-dir data/calibration_synth

# full offline check against ground truth
python selftest_synthetic.py
```

Outputs land in `reports/` (PNG figures + text summaries) and
`data/<dataset>/` (`intrinsics.npz`, `camextrinsics.npz`, `result.npz`).

### Interactive 3D view

```powershell
python visualize_interactive.py --data-dir data/calibration_synth --open
```

This writes `reports/visualization.html`, a self-contained Plotly scene with
the robot at every calibration pose, the checkerboard on the end effector, the
estimated camera + frustum, the ground-truth camera (toggleable), and the
calibration image with detected/predicted corners. Drag to orbit, scroll to
zoom, use the slider / arrow keys / Play button to animate poses, and click an
end-effector or board marker to jump to that pose. Plotly is embedded in the
HTML when it can be downloaded, so the page also works offline; pass `--cdn`
for a much smaller file that needs an internet connection.

A rendered preview is saved as `reports/visualization_preview.png`.

### Verified results (synthetic, 15 views, 0.3 px corner noise)

| Quantity | Value |
|----------|-------|
| PnP reprojection RMSE (step 3) | 0.51 px mean / 1.37 px max |
| Linear hand-eye (`park_numpy`) vs ground truth | 0.29° / 5.35 mm |
| After reprojection refinement | **0.10° / 2.31 mm** |
| FK + hand-eye predicted reprojection (step 5) | 0.47 px mean / 2.92 px max |
| Ray-intersection consistency | 0.35 mm mean / 3.77 mm max |

`selftest_synthetic.py` requires the refined result to stay below
**1° / 5 mm** and prints the linear solvers next to it for comparison.

---

## 4. Folder layout

```
data/
  kinova_gen3_7dof.urdf        # kinematic model (Kortex GEN3_URDF_V12, no meshes)
  calibration/                 # real dataset (images/, joint_angles.txt, intrinsics.npz)
posekit/
  se3.py       # SO(3)/SE(3) helpers, ray triangulation, back-projection
  urdf.py      # URDF parser + forward kinematics
  robot.py     # RobotModel wrapper (fk, fk_many, link_frames)
  camera.py    # intrinsics, checkerboard, detection, solvePnP extrinsics
  handeye.py   # eye-to-hand solver + multi-method comparison
  refine.py    # nonlinear reprojection refinement of X and board-to-EE pose
  io.py        # natural image ordering (1, 2, ..., 10, not 1, 10, 2)
  synthetic.py # synthetic dataset generator (renders the board on the EE)
  plotting.py  # matplotlib reports
step1..step5   # pipeline scripts
visualize_interactive.py  # self-contained Plotly 3D scene -> reports/visualization.html
selftest_synthetic.py
export_for_matlab.py  # optional: dataset + intrinsics back to MATLAB .mat/.zip
reports/       # generated figures / summaries
```

---

## 5. Notes & pitfalls

* **Joint angles** must be the raw encoder values in degrees (`0..360`), like
  `jointAngles.mat`. Row *k* corresponds to image *k*.
* **Image order**: do not use plain `sorted(glob("*.jpg"))`. It yields
  `1, 10, 11, ..., 2`, which silently pairs the wrong joints with the images.
  The pipeline uses `posekit.io.list_images` (natural sort).
* **Board pose on the gripper** is irrelevant for the solve, but the board must
  be rigidly fixed and fully visible (the code drops partial detections, like
  `PartialDetections=false`).
* **Rotation diversity matters more than translation**: poses with < 10° of
  relative rotation contribute little; the solver warns about them.
* **Solver**: the default is a dependency-light Park–Martin implementation
  (`park_numpy`) on `A = B_j B_i^-1`, `B_rel = C_j C_i^-1`. It does not depend
  on OpenCV's eye-to-hand input convention. For `cv2.calibrateHandEye` the
  working eye-to-hand call is `gripper2base = inv(B)` and `target2cam = C`.
* **Refinement**: step 4 then minimizes the actual corner reprojection error
  over `T_camera_to_base` and `T_board_to_end_effector`. On the synthetic
  dataset this improves the error from 0.29° / 5.35 mm to **0.10° / 2.3 mm**
  (0.3 px corner noise). Use `--no-refine` to see the linear result.
* **Method spread is not accuracy**: Tsai/Andreff give ~5.8° / 60 mm on this
  particular robot path, while Park/Horaud/Daniilidis agree with the default.
  Near-parallel rotation axes make some linear solvers ill-conditioned.
* The reference images of the MathWorks example are *not* redistributed here
  (copyright). Use your own captures or the synthetic dataset.

---

## 6. References

* MathWorks – *Estimate Pose of Fixed Camera Relative to Robot Base*
* MathWorks – *What Is Robot Hand-Eye Calibration?*
* R. Tsai, R. Lenz – *A New Technique for Fully Autonomous and Efficient 3D
  Robotics Hand/Eye Calibration*, 1989
* F. Park, B. Martin – *Robot Sensor Calibration: Solving AX = XB on the
  Euclidean Group*, 1994
* K. Daniilidis – *Hand-Eye Calibration Using Dual Quaternions*, 1998

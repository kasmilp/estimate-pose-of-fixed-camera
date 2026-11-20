"""Loading and writing calibration datasets.

A dataset directory has the following layout::

    my_dataset/
      images/
        1.jpg
        2.jpg
        ...
      joint_angles.txt      # one row per image, degrees, N x n_dof
      intrinsics.npz        # K, distortion, image_size (+ board metadata)
      ground_truth.npz      # optional, only for synthetic data
      dataset.json          # optional metadata

Images are sorted naturally (``1, 2, ..., 10``) so that row ``k`` of
``joint_angles.txt`` always belongs to image ``k``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import (
    CameraIntrinsics,
    Checkerboard,
    load_intrinsics,
    save_intrinsics,
)

__all__ = [
    "CalibrationDataset",
    "natural_sort_key",
    "list_images",
    "save_dataset_metadata",
]


def natural_sort_key(path: str | Path) -> list:
    """Sort ``1.jpg, 2.jpg, 10.jpg`` numerically, not lexicographically."""
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def list_images(directory: str | Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exts = {ext.lower() for ext in extensions}
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in exts), key=natural_sort_key)


@dataclass
class CalibrationDataset:
    root: Path
    image_paths: list[Path]
    joint_angles_deg: np.ndarray
    intrinsics: CameraIntrinsics
    board: Checkerboard
    ground_truth: dict[str, np.ndarray] | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.intrinsics.image_size

    @classmethod
    def load(
        cls,
        root: str | Path,
        board: Checkerboard | None = None,
        intrinsics: CameraIntrinsics | None = None,
        require_intrinsics: bool = True,
    ) -> "CalibrationDataset":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {root}")

        images_dir = root / "images"
        image_paths = list_images(images_dir) if images_dir.is_dir() else list_images(root)
        if not image_paths:
            raise FileNotFoundError(f"no images found in {images_dir} or {root}")

        joint_path = root / "joint_angles.txt"
        if joint_path.exists():
            joint_angles = np.loadtxt(joint_path, ndmin=2, dtype=float)
        else:
            npy_path = root / "joint_angles.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"missing joint_angles.txt in {root}")
            joint_angles = np.atleast_2d(np.load(npy_path))

        if len(image_paths) != len(joint_angles):
            raise ValueError(
                f"{len(image_paths)} images but {len(joint_angles)} joint-angle rows; "
                "they must be paired one-to-one"
            )

        loaded_board = board
        loaded_intrinsics = intrinsics
        if loaded_intrinsics is None:
            for name in ("intrinsics.npz", "intrinsics.json"):
                candidate = root / name
                if candidate.exists():
                    loaded_intrinsics, file_board = load_intrinsics(candidate)
                    if loaded_board is None:
                        loaded_board = file_board
                    break

        if loaded_intrinsics is None and require_intrinsics:
            raise FileNotFoundError(
                f"no intrinsics.npz/intrinsics.json in {root}; run the 'intrinsics' command first"
            )

        metadata: dict = {}
        metadata_path = root / "dataset.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if loaded_board is None and "board" in metadata:
            loaded_board = Checkerboard.from_dict(metadata["board"])
        if loaded_board is None:
            loaded_board = Checkerboard()

        if loaded_intrinsics is None:
            # Infer the image size so callers can run intrinsic calibration.
            image = cv2.imread(str(image_paths[0]))
            if image is None:
                raise FileNotFoundError(image_paths[0])
            loaded_intrinsics = CameraIntrinsics(
                K=np.eye(3),
                distortion=np.zeros(5),
                image_size=(image.shape[1], image.shape[0]),
            )

        dataset = cls(
            root=root,
            image_paths=image_paths,
            joint_angles_deg=joint_angles,
            intrinsics=loaded_intrinsics,
            board=loaded_board,
            metadata=metadata,
        )
        dataset.ground_truth = _load_ground_truth(root)
        dataset.validate()
        return dataset

    def validate(self) -> None:
        if len(self.image_paths) != len(self.joint_angles_deg):
            raise ValueError("images and joint angles are not paired")
        if len(self.image_paths) < 3:
            raise ValueError("at least 3 calibration views are required")
        image = cv2.imread(str(self.image_paths[0]))
        if image is None:
            raise FileNotFoundError(self.image_paths[0])
        size = (image.shape[1], image.shape[0])
        if size != self.intrinsics.image_size:
            raise ValueError(
                f"image size {size} does not match intrinsics.image_size {self.intrinsics.image_size}"
            )


def save_dataset_metadata(
    root: str | Path,
    intrinsics: CameraIntrinsics,
    board: Checkerboard,
    joint_angles_deg: np.ndarray,
    ground_truth: dict[str, np.ndarray] | None = None,
    extra: dict | None = None,
) -> None:
    """Write intrinsics, board, joint angles and optional ground truth."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_intrinsics(root / "intrinsics.npz", intrinsics, board)
    np.savetxt(root / "joint_angles.txt", np.asarray(joint_angles_deg, dtype=float), fmt="%.6f")
    if ground_truth is not None:
        np.savez(root / "ground_truth.npz", **{k: np.asarray(v) for k, v in ground_truth.items()})

    metadata = {
        "board": board.to_dict(),
        "intrinsics": intrinsics.to_dict(),
        "views": int(len(joint_angles_deg)),
    }
    if extra:
        metadata.update(extra)
    (root / "dataset.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")


def _load_ground_truth(root: Path) -> dict[str, np.ndarray] | None:
    path = root / "ground_truth.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value

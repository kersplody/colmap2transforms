# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared COLMAP <-> transforms.json helpers."""

from __future__ import annotations

import re
import warnings
from pathlib import PurePosixPath
from typing import Any, Dict, Tuple

import numpy as np


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """Parse a COLMAP camera into nerfstudio-style transforms.json intrinsics."""
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }
    camera_model = getattr(camera, "model_name", camera.model)
    camera_params = camera.params

    if camera_model == "SIMPLE_PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera_model == "PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera_model == "SIMPLE_RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera_model == "RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera_model == "OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["camera_model"] = "OPENCV"
    elif camera_model == "OPENCV_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        out["camera_model"] = "OPENCV_FISHEYE"
    elif camera_model == "SIMPLE_RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        out["camera_model"] = "OPENCV_FISHEYE"
    elif camera_model == "RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0.0
        out["k4"] = 0.0
        out["camera_model"] = "OPENCV_FISHEYE"
    else:
        raise NotImplementedError(f"{camera_model} camera model is not supported")

    return out


def colmap_to_nerfstudio_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    keep_original_world_coordinate: bool = False,
) -> np.ndarray:
    """Convert COLMAP OpenCV world-to-camera pose into nerfstudio-style OpenGL camera-to-world."""
    w2c = np.concatenate([rotation, translation.reshape(3, 1)], axis=1)
    w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1
    if not keep_original_world_coordinate:
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1
    return c2w


def nerfstudio_to_colmap_pose(
    transform_matrix: Any,
    applied_transform: Any = None,
    applied_scale: float = 1.0,
) -> np.ndarray:
    """Convert nerfstudio-style OpenGL camera-to-world pose into COLMAP OpenCV camera-to-world."""
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.concatenate([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected transform_matrix to have shape (4, 4) or (3, 4), got {c2w.shape}")

    c2w[:3, 3] /= applied_scale

    if applied_transform is not None:
        transform = np.asarray(applied_transform, dtype=np.float64)
        if transform.shape == (3, 4):
            transform = np.concatenate([transform, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        if transform.shape != (4, 4):
            raise ValueError(f"Expected applied_transform to have shape (4, 4) or (3, 4), got {transform.shape}")
        c2w = np.linalg.inv(transform) @ c2w

    c2w[0:3, 1:3] *= -1
    return c2w


def normalize_posix_parts(path: str) -> Tuple[str, ...]:
    return tuple(part for part in PurePosixPath(path).parts if part not in ("", "."))


def relative_image_name(file_path: str, image_dir: str) -> str:
    file_parts = normalize_posix_parts(file_path)
    image_dir_parts = normalize_posix_parts(image_dir)
    if image_dir_parts and file_parts[: len(image_dir_parts)] == image_dir_parts:
        relative_parts = file_parts[len(image_dir_parts) :]
        if relative_parts:
            return PurePosixPath(*relative_parts).as_posix()
    return PurePosixPath(*file_parts).as_posix()


def parse_frame_drop_spec(spec: str | None) -> set[int]:
    """Parse a comma-separated frame selection like '1,2,4-5,8-10'."""
    if spec is None or spec.strip() == "":
        return set()

    dropped_frames: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid frame range '{item}': range end must be >= range start")
            dropped_frames.update(range(start, end + 1))
        else:
            dropped_frames.add(int(item))
    return dropped_frames


def extract_frame_number(file_path: str) -> int | None:
    """Extract the trailing frame number from a file name like image_00123.png."""
    name = PurePosixPath(file_path.replace("\\", "/")).name
    stem = PurePosixPath(name).stem
    match = re.search(r"(\d+)(?!.*\d)", stem)
    if match is None:
        return None
    return int(match.group(1))


def warn_for_missing_dropped_frames(requested: set[int], present: set[int]) -> None:
    """Warn only when a drop_frames request does not match any input frames."""
    if not requested:
        return
    matched = requested & present
    if matched:
        return
    preview = ",".join(str(value) for value in sorted(requested)[:10])
    if len(requested) > 10:
        preview = f"{preview},..."
    warnings.warn(f"Requested drop_frames did not match any input frames: {preview}", stacklevel=2)

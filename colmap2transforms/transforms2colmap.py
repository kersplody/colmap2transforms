#!/usr/bin/env python

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

"""Create a COLMAP sparse model from a transforms.json file."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pycolmap

from .common import (
    HelpOnErrorArgumentParser,
    extract_frame_number,
    nerfstudio_to_colmap_pose,
    parse_frame_drop_spec,
    prepare_model_output_dir,
    relative_image_name,
    warn_for_missing_dropped_frames,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _frame_value(frame: Dict[str, Any], meta: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = frame.get(key, meta.get(key, default))
    return float(value)


def _camera_model_from_metadata(frame: Dict[str, Any], meta: Dict[str, Any]) -> str:
    camera_model = frame.get("camera_model", meta.get("camera_model", "OPENCV"))
    if camera_model in ("OPENCV_FISHEYE", "fisheye"):
        return "OPENCV_FISHEYE"
    return "OPENCV"


def _camera_from_frame(camera_id: int, frame: Dict[str, Any], meta: Dict[str, Any]) -> pycolmap.Camera:
    width = int(frame["w"] if "w" in frame else meta["w"])
    height = int(frame["h"] if "h" in frame else meta["h"])
    camera_model = _camera_model_from_metadata(frame, meta)

    if "distortion_params" in frame:
        k1, k2, k3, k4, p1, p2 = [float(x) for x in frame["distortion_params"]]
    elif "distortion_params" in meta:
        k1, k2, k3, k4, p1, p2 = [float(x) for x in meta["distortion_params"]]
    else:
        k1 = _frame_value(frame, meta, "k1")
        k2 = _frame_value(frame, meta, "k2")
        k3 = _frame_value(frame, meta, "k3")
        k4 = _frame_value(frame, meta, "k4")
        p1 = _frame_value(frame, meta, "p1")
        p2 = _frame_value(frame, meta, "p2")

    if camera_model == "OPENCV_FISHEYE":
        params = np.array(
            [
                _frame_value(frame, meta, "fl_x"),
                _frame_value(frame, meta, "fl_y"),
                _frame_value(frame, meta, "cx"),
                _frame_value(frame, meta, "cy"),
                k1,
                k2,
                k3,
                k4,
            ],
            dtype=np.float64,
        )
    else:
        params = np.array(
            [
                _frame_value(frame, meta, "fl_x"),
                _frame_value(frame, meta, "fl_y"),
                _frame_value(frame, meta, "cx"),
                _frame_value(frame, meta, "cy"),
                k1,
                k2,
                p1,
                p2,
            ],
            dtype=np.float64,
        )

    camera = pycolmap.Camera(
        model=getattr(pycolmap.CameraModelId, camera_model),
        width=width,
        height=height,
    )
    camera.camera_id = camera_id
    camera.params = params
    return camera


def create_colmap_data(
    transforms_path: Path,
    output_dir: Path,
    image_dir: str = "./images",
    force_txt: bool = False,
    drop_frames: str | None = None,
    force: bool = False,
) -> None:
    """Create a minimal COLMAP sparse model from a transforms.json file."""
    meta = _load_json(transforms_path)
    dropped_frame_numbers = parse_frame_drop_spec(drop_frames)
    present_frame_numbers = {
        frame_number
        for frame_number in (extract_frame_number(frame["file_path"]) for frame in meta["frames"])
        if frame_number is not None
    }
    warn_for_missing_dropped_frames(dropped_frame_numbers, present_frame_numbers)
    frames = [
        frame
        for frame in meta["frames"]
        if extract_frame_number(frame["file_path"]) not in dropped_frame_numbers
    ]
    if len(frames) == 0:
        raise ValueError("No frames remain after applying drop_frames")

    prepare_model_output_dir(output_dir, force=force)

    applied_transform = meta.get("applied_transform")
    applied_scale = float(meta.get("applied_scale", 1.0))

    cameras: Dict[int, pycolmap.Camera] = {}
    camera_lookup: Dict[Tuple[Any, ...], int] = {}
    reconstruction = pycolmap.Reconstruction()

    next_camera_id = 1
    fallback_image_id = 1
    used_image_ids = set()

    for frame in frames:
        if "fl_x" not in frame and "fl_x" not in meta:
            raise KeyError("Missing intrinsics in transforms.json: expected fl_x/fl_y/cx/cy at top level or per frame")
        if "w" not in frame and "w" not in meta:
            raise KeyError("Missing image width in transforms.json: expected w at top level or per frame")
        if "h" not in frame and "h" not in meta:
            raise KeyError("Missing image height in transforms.json: expected h at top level or per frame")

        camera = _camera_from_frame(next_camera_id, frame, meta)
        camera_key = (camera.model, camera.width, camera.height, tuple(float(x) for x in camera.params))
        camera_id = camera_lookup.get(camera_key)
        if camera_id is None:
            camera_id = next_camera_id
            camera.camera_id = camera_id
            cameras[camera_id] = camera
            reconstruction.add_camera_with_trivial_rig(camera)
            camera_lookup[camera_key] = camera_id
            next_camera_id += 1

        c2w = nerfstudio_to_colmap_pose(
            transform_matrix=frame["transform_matrix"],
            applied_transform=applied_transform,
            applied_scale=applied_scale,
        )
        w2c = np.linalg.inv(c2w)

        image_id = frame.get("colmap_im_id")
        if not isinstance(image_id, int) or image_id in used_image_ids:
            while fallback_image_id in used_image_ids:
                fallback_image_id += 1
            image_id = fallback_image_id
        used_image_ids.add(image_id)

        image_name = relative_image_name(frame["file_path"], image_dir)
        image = pycolmap.Image(
            name=image_name,
            keypoints=np.empty((0, 2), dtype=np.float64),
            camera_id=camera_id,
            image_id=image_id,
        )
        reconstruction.add_image_with_trivial_frame(
            image,
            pycolmap.Rigid3d(np.hstack([w2c[:3, :3], w2c[:3, 3:4]])),
        )

    if force_txt:
        reconstruction.write_text(output_dir)
    else:
        reconstruction.write_binary(output_dir)


@dataclass
class CreateColmap:
    transforms: Path = Path("transforms.json")
    output_dir: Path = Path(".")
    image_dir: str = "./images"
    force_txt: bool = False
    drop_frames: str | None = None
    force: bool = False

    def main(self) -> None:
        transforms_path = self.transforms
        if transforms_path.is_dir():
            transforms_path = transforms_path / "transforms.json"
        create_colmap_data(
            transforms_path=transforms_path,
            output_dir=self.output_dir,
            image_dir=self.image_dir,
            force_txt=self.force_txt,
            drop_frames=self.drop_frames,
            force=self.force,
        )
        print(f"Saved COLMAP sparse model to {self.output_dir}")


def entrypoint() -> None:
    parser = HelpOnErrorArgumentParser(
        description=__doc__,
        epilog="Typical usage:\n  transforms2colmap transforms.json colmap/sparse/0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("transforms_positional", nargs="?", help="Input transforms.json file or its parent directory")
    parser.add_argument("output_dir_positional", nargs="?", help="Output COLMAP sparse model directory")
    parser.add_argument("--transforms", default=None, help="Input transforms.json file or its parent directory")
    parser.add_argument("--output_dir", "--output-dir", default=None, help="Output COLMAP sparse model directory")
    parser.add_argument("--image_dir", "--image-dir", default="./images", help="Prefix stripped from frame file paths")
    parser.add_argument("--txt", action="store_true", help="Write COLMAP text model files instead of binary files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing COLMAP model files in the output directory")
    parser.add_argument(
        "--drop-frames",
        default=None,
        help="Comma-separated frame numbers or ranges to drop based on trailing digits in file names, e.g. 1,2,4-5,8-10",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    transforms = args.transforms if args.transforms is not None else (args.transforms_positional or "transforms.json")
    output_dir = args.output_dir if args.output_dir is not None else (args.output_dir_positional or ".")
    CreateColmap(
        transforms=Path(transforms),
        output_dir=Path(output_dir),
        image_dir=args.image_dir,
        force_txt=args.txt,
        drop_frames=args.drop_frames,
        force=args.force,
    ).main()


if __name__ == "__main__":
    entrypoint()

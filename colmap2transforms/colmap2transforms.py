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

"""Create transforms.json from a COLMAP model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pycolmap

from .common import (
    colmap_to_nerfstudio_pose,
    extract_frame_number,
    parse_colmap_camera_params,
    parse_frame_drop_spec,
    warn_for_missing_dropped_frames,
)


def _load_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    if not ((cameras_bin.exists() and images_bin.exists()) or (cameras_txt.exists() and images_txt.exists())):
        raise FileNotFoundError(
            f"Expected either {cameras_bin} and {images_bin} or {cameras_txt} and {images_txt} to exist"
        )
    return pycolmap.Reconstruction(model_dir)


def create_transforms_data(
    model_dir: Path,
    image_dir: str = "./images",
    keep_original_world_coordinate: bool = False,
    use_single_camera_mode: bool = True,
    drop_frames: str | None = None,
) -> Dict[str, Any]:
    """Create transforms.json data from COLMAP text or binary files."""
    reconstruction = _load_reconstruction(model_dir)
    cam_id_to_camera = reconstruction.cameras
    im_id_to_image = reconstruction.images
    dropped_frame_numbers = parse_frame_drop_spec(drop_frames)
    present_frame_numbers = {
        frame_number
        for frame_number in (extract_frame_number(im_data.name) for im_data in im_id_to_image.values())
        if frame_number is not None
    }
    warn_for_missing_dropped_frames(dropped_frame_numbers, present_frame_numbers)

    if set(cam_id_to_camera.keys()) != {1}:
        use_single_camera_mode = False
        out: Dict[str, Any] = {}
    else:
        out = parse_colmap_camera_params(cam_id_to_camera[1])

    frames = []
    for im_id, im_data in sorted(im_id_to_image.items()):
        frame_number = extract_frame_number(im_data.name)
        if frame_number in dropped_frame_numbers:
            continue

        cam_from_world = im_data.cam_from_world()
        c2w = colmap_to_nerfstudio_pose(
            rotation=cam_from_world.rotation.matrix(),
            translation=cam_from_world.translation,
            keep_original_world_coordinate=keep_original_world_coordinate,
        )

        frame: Dict[str, Any] = {
            "file_path": f"{image_dir.rstrip('/')}/{im_data.name}",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }

        if not use_single_camera_mode:
            frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

        frames.append(frame)

    if not frames:
        raise ValueError("No frames remain after applying drop_frames")

    out["frames"] = frames

    if not keep_original_world_coordinate:
        applied_transform = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
        out["applied_transform"] = applied_transform

    return out


@dataclass
class CreateTransforms:
    model_dir: Path = Path(".")
    output_file: Path = Path(".")
    image_dir: str = "./images"
    keep_original_world_coordinate: bool = False
    use_single_camera_mode: bool = True
    drop_frames: str | None = None

    def main(self) -> None:
        output_file = self.output_file
        if output_file == Path(".") or output_file.is_dir():
            output_file = output_file / "transforms.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        transforms = create_transforms_data(
            model_dir=self.model_dir,
            image_dir=self.image_dir,
            keep_original_world_coordinate=self.keep_original_world_coordinate,
            use_single_camera_mode=self.use_single_camera_mode,
            drop_frames=self.drop_frames,
        )
        output_file.write_text(json.dumps(transforms, indent=4), encoding="utf-8")
        print(f"Saved transforms to {output_file}")


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir_positional", nargs="?", help="COLMAP model directory containing cameras/images as .bin or .txt files")
    parser.add_argument("output_file_positional", nargs="?", help="Output transforms.json file or directory")
    parser.add_argument(
        "--model_dir",
        "--model-dir",
        default=None,
        help="COLMAP model directory containing cameras/images as .bin or .txt files; .bin is preferred",
    )
    parser.add_argument("--output_file", "--output-file", default=None, help="Output transforms.json file or directory")
    parser.add_argument("--image_dir", "--image-dir", default="./images", help="Prefix used for frame file paths")
    parser.add_argument(
        "--drop-frames",
        default=None,
        help="Comma-separated frame numbers or ranges to drop based on trailing digits in file names, e.g. 1,2,4-5,8-10",
    )
    parser.add_argument(
        "--keep_original_world_coordinate",
        "--keep-original-world-coordinate",
        action="store_true",
        help="Keep COLMAP world coordinates instead of applying the nerfstudio z-up transform",
    )
    parser.add_argument(
        "--use_single_camera_mode",
        "--use-single-camera-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write shared camera intrinsics once when possible",
    )
    args = parser.parse_args()
    model_dir = args.model_dir if args.model_dir is not None else (args.model_dir_positional or ".")
    output_file = args.output_file if args.output_file is not None else (args.output_file_positional or ".")
    CreateTransforms(
        model_dir=Path(model_dir),
        output_file=Path(output_file),
        image_dir=args.image_dir,
        keep_original_world_coordinate=args.keep_original_world_coordinate,
        use_single_camera_mode=args.use_single_camera_mode,
        drop_frames=args.drop_frames,
    ).main()


if __name__ == "__main__":
    entrypoint()

#!/usr/bin/env python

"""Create RealityScan-compatible XMP sidecars from a COLMAP sparse model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pycolmap

from .common import HelpOnErrorArgumentParser, ensure_output_file_writable


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


def _rotation_matrix_from_colmap_image(image: pycolmap.Image) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in image.cam_from_world().rotation.matrix())


def _translation_from_colmap_image(image: pycolmap.Image) -> tuple[float, float, float]:
    return tuple(float(value) for value in image.cam_from_world().translation)


def _transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def _mat_vec_mul(
    matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _negate(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-vector[0], -vector[1], -vector[2])


def _format_triplet(values: tuple[float, float, float]) -> str:
    return " ".join(_format_scalar(value) for value in values)


def _format_matrix(matrix: tuple[tuple[float, float, float], ...]) -> str:
    return " ".join(_format_scalar(value) for row in matrix for value in row)


def _format_scalar(value: float) -> str:
    text = f"{value:.15f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    if text:
        return text
    return "0"


def _camera_fx_fy_cx_cy(camera: pycolmap.Camera) -> tuple[float, float, float, float]:
    params = camera.params
    model = str(camera.model_name).upper()
    if model == "SIMPLE_PINHOLE":
        if len(params) < 3:
            raise ValueError(f"Camera {camera.camera_id} SIMPLE_PINHOLE row is missing parameters")
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == "PINHOLE":
        if len(params) < 4:
            raise ValueError(f"Camera {camera.camera_id} PINHOLE row is missing parameters")
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if model == "SIMPLE_RADIAL":
        if len(params) < 4:
            raise ValueError(f"Camera {camera.camera_id} SIMPLE_RADIAL row is missing parameters")
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == "RADIAL":
        if len(params) < 5:
            raise ValueError(f"Camera {camera.camera_id} RADIAL row is missing parameters")
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model in {"OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
        if len(params) < 4:
            raise ValueError(f"Camera {camera.camera_id} {camera.model_name} row is missing parameters")
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported COLMAP camera model {camera.model_name!r}")


def _focal_length_35mm(camera: pycolmap.Camera) -> float:
    fx, fy, _, _ = _camera_fx_fy_cx_cy(camera)
    focal_px = (fx + fy) / 2.0
    sensor_width_mm = 36.0
    return focal_px * sensor_width_mm / float(camera.width)


def _principal_point_offsets(camera: pycolmap.Camera) -> tuple[float, float]:
    _, _, cx, cy = _camera_fx_fy_cx_cy(camera)
    principal_u = (cx - (camera.width / 2.0)) / float(camera.width)
    principal_v = (cy - (camera.height / 2.0)) / float(camera.width)
    return principal_u, principal_v


def _colmap_to_realityscan_position(camera_center: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = camera_center
    return (x, z, -y)


def _colmap_to_realityscan_rotation(
    rotation_world_to_camera: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    axis_transform_t = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    return tuple(
        tuple(sum(rotation_world_to_camera[row][k] * axis_transform_t[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _distortion_payload(camera: pycolmap.Camera) -> tuple[str, str]:
    model = str(camera.model_name).upper()
    params = camera.params
    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return "brown3", "0 0 0 0 0 0"
    if model == "SIMPLE_RADIAL":
        if len(params) < 4:
            raise ValueError(f"Camera {camera.camera_id} SIMPLE_RADIAL row is missing parameters")
        return "brown3", f"{_format_scalar(float(params[3]))} 0 0 0 0 0"
    if model == "RADIAL":
        if len(params) < 5:
            raise ValueError(f"Camera {camera.camera_id} RADIAL row is missing parameters")
        return "brown3", f"{_format_scalar(float(params[3]))} {_format_scalar(float(params[4]))} 0 0 0 0"
    if model == "OPENCV":
        if len(params) < 8:
            raise ValueError(f"Camera {camera.camera_id} OPENCV row is missing parameters")
        return "brown3", " ".join(_format_scalar(float(value)) for value in (params[4], params[5], params[6], params[7], 0.0, 0.0))
    if model == "FULL_OPENCV":
        if len(params) < 12:
            raise ValueError(f"Camera {camera.camera_id} FULL_OPENCV row is missing parameters")
        return (
            "brown3",
            " ".join(_format_scalar(float(value)) for value in (params[4], params[5], params[6], params[7], params[8], params[9])),
        )
    if model == "OPENCV_FISHEYE":
        if len(params) < 8:
            raise ValueError(f"Camera {camera.camera_id} OPENCV_FISHEYE row is missing parameters")
        return (
            "division",
            " ".join(_format_scalar(float(value)) for value in (params[4], params[5], params[6], params[7], 0.0, 0.0)),
        )
    raise ValueError(f"Unsupported COLMAP camera model {camera.model_name!r}")


def build_xmp_bytes(
    *,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    pose_prior: str = "exact",
    coordinates: str = "absolute",
    calibration_prior: str = "exact",
) -> bytes:
    rotation_world_to_camera = _rotation_matrix_from_colmap_image(image)
    rotation_camera_to_world = _transpose(rotation_world_to_camera)
    camera_center = _mat_vec_mul(rotation_camera_to_world, _negate(_translation_from_colmap_image(image)))
    rs_position = _colmap_to_realityscan_position(camera_center)
    rs_rotation = _colmap_to_realityscan_rotation(rotation_world_to_camera)
    focal_length_35mm = _focal_length_35mm(camera)
    principal_u, principal_v = _principal_point_offsets(camera)
    distortion_model, distortion_coefficients = _distortion_payload(camera)
    aspect_ratio = 1.0
    if camera.height != 0:
        fx, fy, _, _ = _camera_fx_fy_cx_cy(camera)
        if fy != 0.0:
            aspect_ratio = fx / fy

    xml = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xcr:Version="3"
       xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
       xcr:CalibrationPrior="{calibration_prior}"
       xcr:DistortionModel="{distortion_model}"
       xcr:FocalLength35mm="{_format_scalar(focal_length_35mm)}" xcr:Skew="0" xcr:AspectRatio="{_format_scalar(aspect_ratio)}"
       xcr:PrincipalPointU="{_format_scalar(principal_u)}" xcr:PrincipalPointV="{_format_scalar(principal_v)}"
       xcr:PosePrior="{pose_prior}"
       xcr:Coordinates="{coordinates}"
       xcr:InMeshing="1" xcr:InTexturing="1">
      <xcr:Rotation>{_format_matrix(rs_rotation)}</xcr:Rotation>
      <xcr:Position>{_format_triplet(rs_position)}</xcr:Position>
      <xcr:DistortionCoeficients>{distortion_coefficients}</xcr:DistortionCoeficients>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""
    return xml.encode("utf-8")


def _resolve_image_path(image_dir: Path, image_name: str, image_ext: str | None) -> Path:
    relative_path = _image_relative_path(image_name)
    if image_ext:
        relative_path = relative_path.with_suffix(image_ext)
    return image_dir / relative_path


def _image_relative_path(image_name: str) -> Path:
    normalized = PurePosixPath(image_name.replace("\\", "/"))
    parts = [part for part in normalized.parts if part not in ("", ".", "/")]
    if parts and parts[0].endswith(":"):
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid image name: {image_name!r}")
    if len(parts) == 1:
        return Path(parts[0])
    return Path(*parts[-1:])


def create_xmp_files(
    model_dir: Path,
    output_dir: Path,
    image_dir: Path,
    pose_prior: str = "exact",
    coordinates: str = "absolute",
    calibration_prior: str = "exact",
    skip_image_check: bool = False,
    image_ext: str | None = None,
    force: bool = False,
) -> int:
    reconstruction = _load_reconstruction(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_count = 0
    for image_id, image in sorted(reconstruction.images.items()):
        del image_id
        try:
            camera = reconstruction.cameras[image.camera_id]
        except KeyError:
            raise KeyError(f"Image {image.name!r} references missing camera id {image.camera_id}")

        source_image_path = _resolve_image_path(image_dir, image.name, image_ext=image_ext)
        target_path = output_dir / _image_relative_path(image.name).with_suffix(".xmp")

        if not skip_image_check and not source_image_path.exists():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_output_file_writable(target_path, force=force)
        target_path.write_bytes(
            build_xmp_bytes(
                image=image,
                camera=camera,
                pose_prior=pose_prior,
                coordinates=coordinates,
                calibration_prior=calibration_prior,
            )
        )
        written_count += 1

    return written_count


@dataclass
class CreateXmp:
    model_dir: Path = Path(".")
    output_dir: Path = Path(".")
    image_dir: Path = Path(".")
    pose_prior: str = "exact"
    coordinates: str = "absolute"
    calibration_prior: str = "exact"
    skip_image_check: bool = False
    image_ext: str | None = None
    force: bool = False

    def main(self) -> None:
        model_dir = self.model_dir
        if self.output_dir != Path("."):
            output_dir = self.output_dir
        elif self.image_dir != Path("."):
            output_dir = self.image_dir
        else:
            output_dir = model_dir.parent

        if self.image_dir != Path("."):
            image_dir = self.image_dir
        else:
            image_dir = output_dir
        image_ext = self.image_ext
        if image_ext is not None and image_ext.strip():
            image_ext = image_ext.strip()
            if not image_ext.startswith("."):
                image_ext = f".{image_ext}"
        else:
            image_ext = None

        written_count = create_xmp_files(
            model_dir=model_dir,
            output_dir=output_dir,
            image_dir=image_dir,
            pose_prior=self.pose_prior,
            coordinates=self.coordinates,
            calibration_prior=self.calibration_prior,
            skip_image_check=self.skip_image_check,
            image_ext=image_ext,
            force=self.force,
        )
        print(f"Wrote {written_count} XMP file(s) to {output_dir}")


def entrypoint() -> None:
    parser = HelpOnErrorArgumentParser(
        description=__doc__,
        epilog="Typical usage:\n  colmap2xmp colmap/sparse/0 images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir_positional", nargs="?", help="COLMAP model directory containing cameras/images as .bin or .txt files")
    parser.add_argument("output_dir_positional", nargs="?", help="Directory where .xmp files will be written")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="COLMAP model directory containing cameras/images as .bin or .txt files; .bin is preferred",
    )
    parser.add_argument("--output-dir", default=None, help="Directory where .xmp files will be written")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing source images; defaults to --output-dir when provided, otherwise the parent of the COLMAP model directory",
    )
    parser.add_argument(
        "--pose-prior",
        choices=("initial", "exact", "locked"),
        default="exact",
        help="RealityScan pose prior mode to encode in the XMP",
    )
    parser.add_argument(
        "--coordinates",
        choices=("absolute", "relative"),
        default="absolute",
        help="RealityScan coordinate mode to encode in the XMP",
    )
    parser.add_argument(
        "--calibration-prior",
        choices=("initial", "exact"),
        default="exact",
        help="RealityScan calibration prior mode",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Allow writing .xmp sidecars even if the source image files are not present",
    )
    parser.add_argument(
        "--image-ext",
        default=None,
        help="Optional image extension override when matching image filenames, for example .jpg",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .xmp files")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    model_dir = args.model_dir if args.model_dir is not None else (args.model_dir_positional or ".")
    output_dir = args.output_dir if args.output_dir is not None else (args.output_dir_positional or ".")
    image_dir = args.image_dir if args.image_dir is not None else "."
    CreateXmp(
        model_dir=Path(model_dir),
        output_dir=Path(output_dir),
        image_dir=Path(image_dir),
        pose_prior=args.pose_prior,
        coordinates=args.coordinates,
        calibration_prior=args.calibration_prior,
        skip_image_check=args.skip_image_check,
        image_ext=args.image_ext,
        force=args.force,
    ).main()


if __name__ == "__main__":
    entrypoint()

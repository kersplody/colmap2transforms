import json
import subprocess
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colmap2transforms.common import extract_frame_number, parse_frame_drop_spec
from colmap2transforms.colmap2transforms import CreateTransforms, create_transforms_data
from colmap2transforms.colmap2xmp import CreateXmp, create_xmp_files
from colmap2transforms.transforms2colmap import CreateColmap, create_colmap_data


def _source_path() -> Path:
    return Path(__file__).with_name("transforms.json")


def _source_data() -> dict:
    return json.loads(_source_path().read_text(encoding="utf-8"))


def _test_data_dir() -> Path:
    return Path(__file__).with_name("test_data")


def _image_name(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def _remaining_frames(source: dict, dropped: set[int]) -> list[dict]:
    return [frame for frame in source["frames"] if extract_frame_number(frame["file_path"]) not in dropped]


def _parse_xmp(path: Path) -> tuple[dict[str, str], dict[str, list[float]]]:
    namespaces = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "xcr": "http://www.capturingreality.com/ns/xcr/1.1#",
    }
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    description = root.find(".//rdf:Description", namespaces)
    if description is None:
        raise AssertionError(f"Missing rdf:Description in {path}")

    attrs = {
        key.split("}", 1)[1]: value
        for key, value in description.attrib.items()
        if key.startswith("{http://www.capturingreality.com/ns/xcr/1.1#}")
    }
    vectors: dict[str, list[float]] = {}
    for tag in ("Rotation", "Position", "DistortionCoeficients"):
        node = description.find(f"xcr:{tag}", namespaces)
        if node is None or node.text is None:
            raise AssertionError(f"Missing xcr:{tag} in {path}")
        vectors[tag] = [float(value) for value in node.text.split()]
    return attrs, vectors


def _assert_xmp_matches(testcase: unittest.TestCase, actual_path: Path, expected_path: Path) -> None:
    actual_attrs, actual_vectors = _parse_xmp(actual_path)
    expected_attrs, expected_vectors = _parse_xmp(expected_path)

    float_attrs = {"FocalLength35mm", "Skew", "AspectRatio", "PrincipalPointU", "PrincipalPointV"}
    testcase.assertEqual(set(actual_attrs), set(expected_attrs))
    for key, expected_value in expected_attrs.items():
        actual_value = actual_attrs[key]
        if key in float_attrs:
            testcase.assertAlmostEqual(float(actual_value), float(expected_value), places=9)
        else:
            testcase.assertEqual(actual_value, expected_value)

    testcase.assertEqual(set(actual_vectors), set(expected_vectors))
    for key, expected_values in expected_vectors.items():
        actual_values = actual_vectors[key]
        testcase.assertEqual(len(actual_values), len(expected_values))
        for actual_value, expected_value in zip(actual_values, expected_values):
            testcase.assertAlmostEqual(actual_value, expected_value, places=9)


def _assert_round_trip_matches(
    testcase: unittest.TestCase,
    source: dict,
    round_tripped: dict,
) -> None:
    testcase.assertEqual(len(round_tripped["frames"]), len(source["frames"]))
    testcase.assertNotIn("applied_transform", round_tripped)

    for index, (expected_frame, actual_frame) in enumerate(zip(source["frames"], round_tripped["frames"]), start=1):
        testcase.assertEqual(actual_frame["colmap_im_id"], index)
        testcase.assertEqual(_image_name(actual_frame["file_path"]), _image_name(expected_frame["file_path"]))
        testcase.assertEqual(actual_frame["w"], expected_frame["w"])
        testcase.assertEqual(actual_frame["h"], expected_frame["h"])
        testcase.assertEqual(actual_frame["camera_model"], "OPENCV")

        for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2"):
            testcase.assertAlmostEqual(actual_frame[key], expected_frame[key], places=6)

        testcase.assertEqual(actual_frame["p1"], 0.0)
        testcase.assertEqual(actual_frame["p2"], 0.0)
        np.testing.assert_allclose(actual_frame["transform_matrix"], expected_frame["transform_matrix"], atol=1e-9)


class RoundTripTest(unittest.TestCase):
    def test_cli_no_args_prints_help(self) -> None:
        expected_examples = {
            "colmap2transforms.colmap2transforms": "colmap2transforms colmap/sparse/0 transforms.json",
            "colmap2transforms.transforms2colmap": "transforms2colmap transforms.json colmap/sparse/0",
            "colmap2transforms.colmap2xmp": "colmap2xmp colmap/sparse/0 --image-dir images",
        }
        for module_name in (
            "colmap2transforms.colmap2transforms",
            "colmap2transforms.transforms2colmap",
            "colmap2transforms.colmap2xmp",
        ):
            result = subprocess.run(
                [sys.executable, "-m", module_name],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage:", result.stdout)
            self.assertIn(expected_examples[module_name], result.stdout)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_bad_args_print_help(self) -> None:
        for module_name in (
            "colmap2transforms.colmap2transforms",
            "colmap2transforms.transforms2colmap",
            "colmap2transforms.colmap2xmp",
        ):
            result = subprocess.run(
                [sys.executable, "-m", module_name, "--definitely-not-a-real-flag"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stdout)
            self.assertIn("error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_legacy_flag_spellings(self) -> None:
        cases = [
            ("colmap2transforms.colmap2transforms", "--model_dir"),
            ("colmap2transforms.colmap2transforms", "--output_file"),
            ("colmap2transforms.colmap2transforms", "--image_dir"),
            ("colmap2transforms.colmap2transforms", "--keep_original_world_coordinate"),
            ("colmap2transforms.colmap2transforms", "--use_single_camera_mode"),
            ("colmap2transforms.colmap2transforms", "--createPly"),
            ("colmap2transforms.transforms2colmap", "--output_dir"),
            ("colmap2transforms.transforms2colmap", "--image_dir"),
        ]
        for module_name, legacy_flag in cases:
            result = subprocess.run(
                [sys.executable, "-m", module_name, legacy_flag],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(legacy_flag, result.stderr)

    def test_colmap2transforms_refuses_to_overwrite_without_force(self) -> None:
        source_path = _source_path()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            output_file = temp_dir_path / "transforms.json"
            create_colmap_data(source_path, model_dir)
            output_file.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                CreateTransforms(model_dir=model_dir, output_file=output_file).main()

            CreateTransforms(model_dir=model_dir, output_file=output_file, force=True).main()
            self.assertIn('"frames"', output_file.read_text(encoding="utf-8"))

    def test_colmap2transforms_create_ply_refuses_to_overwrite_without_force(self) -> None:
        source_path = _source_path()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            output_file = temp_dir_path / "transforms.json"
            ply_file = temp_dir_path / "sparse_pc.ply"
            create_colmap_data(source_path, model_dir)
            ply_file.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                CreateTransforms(model_dir=model_dir, output_file=output_file, create_ply="sparse_pc.ply").main()

            CreateTransforms(model_dir=model_dir, output_file=output_file, create_ply="sparse_pc.ply", force=True).main()
            ply_bytes = ply_file.read_bytes()
            self.assertIn(b"format binary_little_endian 1.0", ply_bytes)
            self.assertIn(b"element vertex 0", ply_bytes)

    def test_transforms2colmap_refuses_to_overwrite_without_force(self) -> None:
        source_path = _source_path()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sparse"
            create_colmap_data(source_path, model_dir)

            with self.assertRaises(FileExistsError):
                CreateColmap(transforms=source_path, output_dir=model_dir).main()

            CreateColmap(transforms=source_path, output_dir=model_dir, force_txt=True, force=True).main()
            self.assertTrue((model_dir / "cameras.txt").exists())
            self.assertFalse((model_dir / "cameras.bin").exists())

    def test_colmap2xmp_refuses_to_overwrite_without_force(self) -> None:
        source_path = _source_path()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            image_dir = temp_dir_path / "images"
            output_dir = temp_dir_path / "xmp"

            create_colmap_data(source_path, model_dir)
            image_dir.mkdir()
            output_dir.mkdir()
            existing = output_dir / "frame_00001.xmp"
            existing.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                CreateXmp(model_dir=model_dir, image_dir=image_dir, output_dir=output_dir, skip_image_check=True).main()

            CreateXmp(model_dir=model_dir, image_dir=image_dir, output_dir=output_dir, skip_image_check=True, force=True).main()
            self.assertIn("x:xmpmeta", existing.read_text(encoding="utf-8"))

    def test_drop_frame_helpers_match_zero_padded_names(self) -> None:
        self.assertEqual(parse_frame_drop_spec("1,2,4-5"), {1, 2, 4, 5})
        self.assertEqual(extract_frame_number("images_00001.png"), 1)
        self.assertEqual(extract_frame_number("nested/path/frame01475.png"), 1475)

    def test_transforms_round_trip_from_binary_model(self) -> None:
        source_path = _source_path()
        source = _source_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"

            create_colmap_data(source_path, model_dir)

            self.assertTrue((model_dir / "cameras.bin").exists())
            self.assertTrue((model_dir / "images.bin").exists())
            self.assertTrue((model_dir / "points3D.bin").exists())

            round_tripped = create_transforms_data(model_dir, keep_original_world_coordinate=True)

        _assert_round_trip_matches(self, source, round_tripped)

    def test_transforms_round_trip_from_text_model(self) -> None:
        source_path = _source_path()
        source = _source_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"

            create_colmap_data(source_path, model_dir, force_txt=True)

            self.assertTrue((model_dir / "cameras.txt").exists())
            self.assertTrue((model_dir / "images.txt").exists())
            self.assertTrue((model_dir / "points3D.txt").exists())

            round_tripped = create_transforms_data(model_dir, keep_original_world_coordinate=True)

        _assert_round_trip_matches(self, source, round_tripped)

    def test_binary_model_is_preferred_over_text_model(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"

            create_colmap_data(source_path, model_dir)
            (model_dir / "cameras.txt").write_text("not a valid camera model\n", encoding="utf-8")
            (model_dir / "images.txt").write_text("not a valid image model\n", encoding="utf-8")

            round_tripped = create_transforms_data(model_dir, keep_original_world_coordinate=True)

        self.assertEqual(len(round_tripped["frames"]), 1464)

    def test_xmp_generation_prefers_binary_model(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            output_dir = temp_dir_path / "xmp"
            image_dir = temp_dir_path / "images"

            create_colmap_data(source_path, model_dir)
            (model_dir / "cameras.txt").write_text("not a valid camera model\n", encoding="utf-8")
            (model_dir / "images.txt").write_text("not a valid image model\n", encoding="utf-8")

            written_count = create_xmp_files(
                model_dir=model_dir,
                output_dir=output_dir,
                image_dir=image_dir,
                skip_image_check=True,
            )

            self.assertEqual(written_count, 1464)
            self.assertTrue((output_dir / "frame_00001.xmp").exists())
            self.assertIn("xcr:Rotation", (output_dir / "frame_00001.xmp").read_text(encoding="utf-8"))

    def test_colmap2xmp_matches_sample_xmp_for_reduced_binary_model(self) -> None:
        model_dir = _test_data_dir()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            output_dir = temp_dir_path / "xmp"
            image_dir = temp_dir_path / "images"

            written_count = create_xmp_files(
                model_dir=model_dir,
                output_dir=output_dir,
                image_dir=image_dir,
                skip_image_check=True,
            )

            self.assertEqual(written_count, 2)
            for name in ("frame_00001.xmp", "frame_00002.xmp"):
                _assert_xmp_matches(
                    self,
                    output_dir / name,
                    _test_data_dir() / name,
                )

    def test_missing_drop_frames_warns_for_transforms_to_colmap(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sparse"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                create_colmap_data(source_path, model_dir, drop_frames="999999")

        self.assertTrue(any("Requested drop_frames did not match any input frames: 999999" in str(item.message) for item in caught))

    def test_missing_drop_frames_warns_for_colmap_to_transforms(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            create_colmap_data(source_path, model_dir)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                create_transforms_data(model_dir, keep_original_world_coordinate=True, drop_frames="999999")

        self.assertTrue(any("Requested drop_frames did not match any input frames: 999999" in str(item.message) for item in caught))

    def test_partial_range_drop_does_not_warn(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sparse"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                create_colmap_data(source_path, model_dir, drop_frames="1400-2000")

        self.assertEqual(caught, [])

    def test_drop_frames_in_transforms_to_colmap(self) -> None:
        source_path = _source_path()
        source = _source_data()
        dropped = parse_frame_drop_spec("1,2,4-5,8-10,100,1524")
        expected_frames = _remaining_frames(source, dropped)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sparse"
            create_colmap_data(source_path, model_dir, drop_frames="1,2,4-5,8-10,100,1524")
            round_tripped = create_transforms_data(model_dir, keep_original_world_coordinate=True)

        self.assertEqual(len(round_tripped["frames"]), len(expected_frames))
        self.assertEqual(
            [_image_name(frame["file_path"]) for frame in round_tripped["frames"]],
            [_image_name(frame["file_path"]) for frame in expected_frames],
        )

    def test_drop_frames_in_colmap_to_transforms(self) -> None:
        source_path = _source_path()
        source = _source_data()
        dropped = parse_frame_drop_spec("1,2,4-5,8-10,100,1524")
        expected_frames = _remaining_frames(source, dropped)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sparse"
            create_colmap_data(source_path, model_dir)
            round_tripped = create_transforms_data(
                model_dir,
                keep_original_world_coordinate=True,
                drop_frames="1,2,4-5,8-10,100,1524",
            )

        self.assertEqual(len(round_tripped["frames"]), len(expected_frames))
        self.assertEqual(
            [_image_name(frame["file_path"]) for frame in round_tripped["frames"]],
            [_image_name(frame["file_path"]) for frame in expected_frames],
        )

    def test_colmap2transforms_create_ply_writes_file_and_metadata(self) -> None:
        source_path = _source_path()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            model_dir = temp_dir_path / "sparse"
            output_file = temp_dir_path / "nested" / "transforms.json"
            ply_file = output_file.parent / "sparse_pc.ply"

            create_colmap_data(source_path, model_dir)
            CreateTransforms(model_dir=model_dir, output_file=output_file, create_ply="sparse_pc.ply").main()

            transforms = json.loads(output_file.read_text(encoding="utf-8"))
            ply_bytes = ply_file.read_bytes()
            self.assertEqual(transforms["ply_file_path"], "sparse_pc.ply")
            self.assertTrue(ply_file.exists())
            self.assertIn(b"format binary_little_endian 1.0", ply_bytes)
            self.assertIn(b"element vertex 0", ply_bytes)


if __name__ == "__main__":
    unittest.main()

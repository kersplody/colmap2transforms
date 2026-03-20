# colmap2transforms

Utilities for converting between COLMAP sparse models and `transforms.json`.

The package installs two CLI tools:

- `colmap2transforms`
- `transforms2colmap`

## Install

```bash
pip install .
```

For test dependencies:

```bash
pip install .[test]
```

This package currently depends on `pycolmap` and `numpy`.

## Commands

### `colmap2transforms`

Create a `transforms.json` file from a COLMAP model directory.

The input model may be text or binary:

- `cameras.bin` / `images.bin`
- `cameras.txt` / `images.txt`

If both exist, binary is preferred.

Examples:

```bash
colmap2transforms sparse/ transforms.json
```

```bash
colmap2transforms --image-dir ./images sparse/ transforms.json
```

```bash
colmap2transforms --keep-original-world-coordinate sparse/ transforms.json
```

```bash
colmap2transforms --drop-frames=1,2,4-5,8-10,100,1524 sparse/ transforms.json
```

```bash
colmap2transforms --create-ply sparse_pc.ply sparse/ transforms.json
```

By default this command refuses to overwrite an existing output file. Use `--force` to replace it.

When `--create-ply` is provided, the command also writes a nerfstudio-compatible ASCII sparse point cloud PLY from COLMAP's `points3D` data and records `ply_file_path` in `transforms.json`.

### `transforms2colmap`

Create a COLMAP sparse model from a `transforms.json` file.

By default this writes a binary model. Use `--txt` to force text output.

Examples:

```bash
transforms2colmap transforms.json sparse/
```

```bash
transforms2colmap --txt transforms.json sparse/
```

```bash
transforms2colmap --image-dir ./images transforms.json sparse/
```

```bash
transforms2colmap --drop-frames=1,2,4-5,8-10,100,1524 transforms.json sparse/
```

By default this command refuses to overwrite existing COLMAP model files in the output directory. Use `--force` to replace them.

## Drop Frames

Both commands support:

```bash
--drop-frames=1,2,4-5,8-10
```

Frame numbers are extracted from the trailing digits in the filename stem. For example:

- `images_00001.png` -> `1`
- `frame01475.png` -> `1475`

If a drop spec matches no input frames, the tool warns. Partial range misses do not warn.

## Testing

Run the packaged test suite with:

```bash
pytest -q
```

Or run the local script directly:

```bash
python3 test/test_round_trip.py
```

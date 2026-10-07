# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.dataset on a tiny dataset_4dgs/ built the way cumuli's
build_flipbook_4dgs_dataset.py builds one: extensionless file_path entries,
per-frame intrinsics, RGBA images, and a points3d.ply with uchar colour and
a float time property."""

import json
import struct

import numpy as np
import pytest

from cumuli_core.dataset import Dataset4D, read_ply_vertices

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

W, H, FPS = 8, 6, 24.0
C2W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]


def rows(subdir, labels, frames, namer):
    out = []
    for label in labels:
        for i in frames:
            out.append({"file_path": f"{subdir}/cam{label}/{namer(label, i)}",
                        "camera_label": label, "time": i / FPS,
                        "fl_x": 10.0, "fl_y": 10.0, "cx": W / 2, "cy": H / 2,
                        "w": W, "h": H, "transform_matrix": C2W})
    return out


def write_points(path, pts, rgb, times):
    """Exactly the layout of build_4dgs_dataset.write_ply_with_time."""
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(pts)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property float time\nend_header\n')
    rec = struct.Struct('<fffBBBf')
    with open(path, 'wb') as fp:
        fp.write(header.encode('ascii'))
        for p, c, t in zip(pts, rgb, times):
            fp.write(rec.pack(p[0], p[1], p[2], c[0], c[1], c[2], t))


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "dataset_4dgs"
    frame_name = lambda label, i: f"frame_{i + 1:05d}"
    eval_name = lambda label, i: f"cam{label}_frame_{i + 1:05d}"
    rng = np.random.default_rng(0)
    for label in ("00", "01"):
        (root / "realcams" / f"cam{label}").mkdir(parents=True)
        for i in range(3):
            rgba = rng.integers(0, 256, (H, W, 4), dtype=np.uint8)
            Image.fromarray(rgba, "RGBA").save(
                root / "realcams" / f"cam{label}" / f"{frame_name(label, i)}.png")
    (root / "evalcams" / "came0").mkdir(parents=True)
    (root / "eval_gt_flat").mkdir()
    for i in range(3):
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        Image.fromarray(rgb, "RGB").save(
            root / "evalcams" / "came0" / f"{eval_name('e0', i)}.png")
        Image.fromarray(rgb, "RGB").save(root / "eval_gt_flat" / f"{eval_name('e0', i)}.png")
    (root / "transforms_train.json").write_text(json.dumps(
        {"camera_model": "OPENCV", "frames": rows("realcams", ["00", "01"], range(3), frame_name)}))
    (root / "transforms_test.json").write_text(json.dumps(
        {"camera_model": "OPENCV", "frames": rows("evalcams", ["e0"], range(3), eval_name)}))

    pts = rng.normal(size=(20, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (20, 3), dtype=np.uint8)
    times = np.repeat(np.arange(4) / FPS, 5).astype(np.float32)
    write_points(root / "points3d.ply", pts, rgb, times)
    return root, pts, rgb, times


def test_cameras_and_time_span(dataset_dir):
    root, *_ = dataset_dir
    ds = Dataset4D(root)
    assert len(ds.train_cameras) == 6 and len(ds.test_cameras) == 3
    assert ds.time_min == 0.0 and ds.time_max == pytest.approx(2 / FPS)
    assert ds.times == pytest.approx([0.0, 1 / FPS, 2 / FPS])
    assert ds.train_cameras[0].label == "00"


def test_load_image_is_rgba_float_in_unit_range(dataset_dir):
    root, *_ = dataset_dir
    ds = Dataset4D(root)
    cam = ds.train_cameras[4]
    img = ds.load_image(cam)
    assert img.dtype == np.float32 and img.shape == (H, W, 4)
    raw = np.asarray(Image.open(root / "realcams" / "cam01" / "frame_00002.png"))
    np.testing.assert_allclose(img, raw / 255.0, atol=1e-7)


def test_rgb_image_gets_opaque_alpha(dataset_dir):
    root, *_ = dataset_dir
    ds = Dataset4D(root)
    img = ds.load_image(ds.test_cameras[1])
    assert img.shape == (H, W, 4)
    assert np.all(img[..., 3] == 1.0)


def test_eval_gt_lookup(dataset_dir):
    root, *_ = dataset_dir
    ds = Dataset4D(root)
    assert ds.gt_path(ds.test_cameras[0]).endswith("came0_frame_00001.png")
    assert ds.gt_path(ds.train_cameras[0]) is None


def test_size_mismatch_is_an_error(dataset_dir):
    root, *_ = dataset_dir
    ds = Dataset4D(root, downscale=2)
    with pytest.raises(ValueError, match="transforms say"):
        ds.load_image(ds.train_cameras[0])


def test_load_points(dataset_dir):
    root, pts, rgb, times = dataset_dir
    p = Dataset4D(root).load_points()
    np.testing.assert_array_equal(p["xyz"], pts)
    np.testing.assert_allclose(p["rgb"], rgb / 255.0, atol=1e-7)
    np.testing.assert_array_equal(p["time"], times)
    assert set(p) == {"xyz", "rgb", "time"}


def test_ascii_ply_with_extra_properties(tmp_path):
    path = tmp_path / "a.ply"
    path.write_text("ply\nformat ascii 1.0\ncomment hi\nelement vertex 2\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property float nx\nproperty float ny\nproperty float nz\n"
                    "property float red\nproperty float green\nproperty float blue\n"
                    "property float opacity\n"
                    "element face 0\nproperty list uchar int vertex_indices\n"
                    "end_header\n"
                    "1 2 3 0 0 1 0.5 0.25 1 0.9\n4 5 6 0 1 0 0 0 0 0.1\n")
    info, cols = read_ply_vertices(path)
    assert info["format"] == "ascii" and info["comments"] == ["hi"]
    np.testing.assert_allclose(cols["y"], [2, 5])
    root = tmp_path / "ds"
    root.mkdir()
    (root / "transforms_train.json").write_text(json.dumps({"frames": []}))
    p = Dataset4D(root).load_points(path)
    np.testing.assert_allclose(p["rgb"][0], [0.5, 0.25, 1.0])
    np.testing.assert_allclose(p["normals"][1], [0, 1, 0])
    np.testing.assert_allclose(p["opacity"], [0.9, 0.1], rtol=1e-6)


def test_truncated_binary_ply_raises(dataset_dir):
    root, *_ = dataset_dir
    raw = (root / "points3d.ply").read_bytes()
    (root / "points3d.ply").write_bytes(raw[:-10])
    with pytest.raises(ValueError, match="truncated"):
        Dataset4D(root).load_points()


def test_not_a_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dataset4D(tmp_path)

# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.cameras.  The first block is carried over from cumuli's
test_eval_report.py.  Two silent mis-scoring bugs lived in this loader:
image size was read only from the top level of the transforms file, so a
dataset with per-camera intrinsics could not trip the size check, and a
wrong downscale then rendered at the wrong scale."""

import json
import os

import numpy as np
import pytest

from cumuli_core.cameras import Camera, group_by_time, load_transforms

IDENTITY_AT_Z3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]


def per_frame_transforms(tmp_path, n=2, w=960, h=840, fl=853.33, times=(0.0, 0.5)):
    """Shaped like build_flipbook_4dgs_dataset.py output: every camera
    carries its own intrinsics, and there is no top-level block."""
    frames = []
    for i in range(n):
        frames.append({
            "file_path": f"evalcams/came{i:02d}/came{i:02d}_frame_00001",
            "camera_label": f"e{i:02d}",
            "time": times[i % len(times)],
            "fl_x": fl, "fl_y": fl, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
            "transform_matrix": IDENTITY_AT_Z3,
        })
    path = tmp_path / "transforms_test.json"
    path.write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}))
    return path


def test_image_size_is_read_per_frame_when_there_is_no_global_block(tmp_path):
    cams = load_transforms(per_frame_transforms(tmp_path))
    assert [c.width for c in cams] == [960, 960]
    assert [c.height for c in cams] == [840, 840]


def test_downscale_applies_to_the_per_frame_size_too(tmp_path):
    cams = load_transforms(per_frame_transforms(tmp_path), downscale=2)
    assert cams[0].width == 480 and cams[0].height == 420
    assert cams[0].K[0, 0] == pytest.approx(853.33 / 2)
    assert cams[0].K[0, 2] == pytest.approx(240.0)


def test_a_global_intrinsics_block_still_works(tmp_path):
    """The n3v-style layout: one intrinsics block for every camera."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "w": 1024, "h": 768, "fl_x": 800.0, "fl_y": 800.0,
        "cx": 512.0, "cy": 384.0,
        "frames": [{"file_path": "a", "time": 0.0,
                    "transform_matrix": IDENTITY_AT_Z3}]}))
    cams = load_transforms(path, downscale=2)
    assert cams[0].width == 512 and cams[0].height == 384
    assert cams[0].K[0, 0] == pytest.approx(400.0)
    assert cams[0].label is None


def test_missing_size_is_none(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "fl_x": 800.0, "fl_y": 800.0, "cx": 512.0, "cy": 384.0,
        "frames": [{"file_path": "a", "transform_matrix": IDENTITY_AT_Z3}]}))
    cam = load_transforms(path)[0]
    assert cam.width is None and cam.height is None
    assert cam.time == 0.0


def test_every_selects_a_subset(tmp_path):
    cams = load_transforms(per_frame_transforms(tmp_path, n=6), every=3)
    assert len(cams) == 2


def test_camera_names_come_from_the_file_path_basename(tmp_path):
    """Ground truth is found by this name, so it has to be unique per view."""
    cams = load_transforms(per_frame_transforms(tmp_path))
    assert cams[0].name == "came00_frame_00001"
    assert cams[1].name == "came01_frame_00001"


def test_image_path_resolves_against_the_json_directory(tmp_path):
    cams = load_transforms(per_frame_transforms(tmp_path))
    assert cams[0].image_path == os.path.join(
        str(tmp_path), "evalcams", "came00", "came00_frame_00001")
    assert cams[1].label == "e01"


def test_opengl_pose_becomes_opencv_world_to_camera(tmp_path):
    """A GL camera at z = 3 looking down -z sees the origin straight ahead.
    In OpenCV that point has positive depth 3 and projects to the centre."""
    cam = load_transforms(per_frame_transforms(tmp_path))[0]
    origin_cam = cam.w2c @ np.array([0.0, 0.0, 0.0, 1.0])
    assert origin_cam[:3] == pytest.approx([0.0, 0.0, 3.0])
    up = cam.w2c @ np.array([0.0, 1.0, 0.0, 1.0])
    assert up[1] < 0                       # world +y is image up, so camera -y
    uvw = cam.K @ origin_cam[:3]
    assert uvw[:2] / uvw[2] == pytest.approx([480.0, 420.0])
    np.testing.assert_allclose(cam.center, [0.0, 0.0, 3.0])


def test_rotated_pose_matches_the_reference_formula(tmp_path):
    """Same flip-and-invert eval_render.py always used, on a general pose."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    c2w = np.eye(4)
    c2w[:3, :3] = q
    c2w[:3, 3] = rng.normal(size=3)
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"fl_x": 1, "fl_y": 1, "cx": 0, "cy": 0, "frames": [
        {"file_path": "x", "transform_matrix": c2w.tolist()}]}))
    ref = c2w.copy()
    ref[:3, 1:3] *= -1
    np.testing.assert_allclose(load_transforms(path)[0].w2c, np.linalg.inv(ref),
                               atol=1e-12)


def test_group_by_time_sorts_times_and_keeps_camera_order(tmp_path):
    cams = load_transforms(per_frame_transforms(tmp_path, n=6, times=(0.5, 0.0, 0.25)))
    groups = group_by_time(cams)
    assert list(groups) == [0.0, 0.25, 0.5]
    assert [c.label for c in groups[0.5]] == ["e00", "e03"]
    assert all(isinstance(c, Camera) for g in groups.values() for c in g)

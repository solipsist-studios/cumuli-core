# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.shutter: raw camera models against OpenCV, transforms parsing,
and the per-Gaussian rolling-shutter times."""

import json

import numpy as np
import pytest

from cumuli_core.cameras import Camera, load_transforms
from cumuli_core.shutter import (RawCamera, distort_normalized, pixel_offsets, row_fraction,
                                 row_fraction_normalized, sensor_of)

cv2 = pytest.importorskip("cv2")

IDENTITY_AT_Z3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]
RAW_K = [[1400.0, 0.0, 1925.0], [0.0, 1395.0, 1070.0], [0.0, 0.0, 1.0]]
FISHEYE = RawCamera.from_json({"model": "OPENCV_FISHEYE", "K": RAW_K,
                               "dist": [0.05, -0.02, 0.004, -0.001], "w": 3840, "h": 2160})


def grid(n=25, extent=1.2):
    x, y = np.meshgrid(np.linspace(-extent, extent, n), np.linspace(-0.8 * extent, 0.8 * extent, n))
    return x.ravel(), y.ravel()


def write(tmp_path, rows):
    p = tmp_path / "transforms.json"
    p.write_text(json.dumps({"frames": rows}))
    return p


def row(**extra):
    return {"file_path": "a", "time": 0.5, "fl_x": 500.0, "fl_y": 500.0, "cx": 320.0, "cy": 240.0,
            "w": 640, "h": 480, "transform_matrix": IDENTITY_AT_Z3, **extra}


def test_old_files_parse_as_global_shutter(tmp_path):
    cam, = load_transforms(write(tmp_path, [row()]))
    assert cam.readout_s == 0.0 and cam.raw_camera is None and not cam.rolling_shutter
    plain = Camera(name="c", image_path="c", w2c=np.eye(4), K=np.eye(3), width=1, height=1, time=0.0)
    assert plain.readout_s == 0.0 and plain.raw_camera is None


def test_readout_and_raw_camera_parse(tmp_path):
    raw = {"model": "OPENCV_FISHEYE", "K": RAW_K, "dist": [0.05, -0.02, 0.004, -0.001], "w": 3840, "h": 2160}
    cam, = load_transforms(write(tmp_path, [row(readout_s=0.0185, raw_camera=raw)]), downscale=2)
    assert cam.readout_s == pytest.approx(0.0185) and cam.rolling_shutter
    rc = cam.raw_camera
    assert rc.model == "OPENCV_FISHEYE" and (rc.width, rc.height) == (3840, 2160)
    assert rc.dist == (0.05, -0.02, 0.004, -0.001)
    np.testing.assert_array_equal(rc.K, np.asarray(RAW_K))   # raw K is not downscaled
    assert rc.to_json() == raw
    # readout 0 / null is a global shutter
    cam, = load_transforms(write(tmp_path, [row(readout_s=None)]))
    assert cam.readout_s == 0.0


@pytest.mark.parametrize("bad", [{"model": "KANNALA", "K": RAW_K, "dist": [], "w": 1, "h": 1},
                                 {"model": "OPENCV_FISHEYE", "K": RAW_K, "dist": [0.1], "w": 1, "h": 1},
                                 {"model": "OPENCV", "K": [[1, 0], [0, 1]], "dist": [0, 0, 0, 0], "w": 1, "h": 1}])
def test_bad_raw_camera_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        load_transforms(write(tmp_path, [row(readout_s=0.01, raw_camera=bad)]))


def test_fisheye_matches_opencv():
    x, y = grid()
    pts = np.stack([x, y], -1)[None].astype(np.float64)
    want = cv2.fisheye.distortPoints(pts, np.asarray(RAW_K), np.asarray(FISHEYE.dist))[0]
    u, v = distort_normalized(x, y, FISHEYE)
    np.testing.assert_allclose(np.stack([u, v], -1), want, atol=1e-6)
    # the optical axis maps to the principal point
    u0, v0 = distort_normalized(np.zeros(1), np.zeros(1), FISHEYE)
    assert (u0[0], v0[0]) == pytest.approx((1925.0, 1070.0))


@pytest.mark.parametrize("dist", [[-0.28, 0.07, 0.001, -0.0007],
                                  [-0.28, 0.07, 0.001, -0.0007, -0.01],
                                  [0.3, -0.1, 0.0005, 0.0002, 0.02, 0.6, -0.05, 0.03]])
def test_opencv_brown_conrady_matches_projectpoints(dist):
    raw = RawCamera.from_json({"model": "OPENCV", "K": RAW_K, "dist": dist, "w": 3840, "h": 2160})
    x, y = grid(extent=0.7)
    obj = np.stack([x, y, np.ones_like(x)], -1)[:, None, :]
    want, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), np.asarray(RAW_K), np.asarray(dist))
    u, v = distort_normalized(x, y, raw)
    np.testing.assert_allclose(np.stack([u, v], -1), want[:, 0], atol=1e-6)


def test_row_fraction_goes_through_the_image_intrinsics():
    """An undistorted image at any scale and crop maps to the same raw rows: the
    pixel goes through its own pinhole K to normalized coordinates first."""
    K_full = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
    K_half = K_full.copy()
    K_half[:2] /= 2
    u = np.array([100.0, 960.0, 1800.0])
    v = np.array([30.0, 540.0, 1000.0])
    a = row_fraction(u, v, K_full, FISHEYE)
    b = row_fraction(u / 2, v / 2, K_half, FISHEYE)
    np.testing.assert_allclose(a, b, atol=1e-12)
    xn, yn = (u - 960.0) / 1000.0, (v - 540.0) / 1000.0
    np.testing.assert_allclose(a, row_fraction_normalized(xn, yn, FISHEYE), atol=1e-12)
    # the raw middle line is fraction 0: the principal point's row 1070 of 2160 is just below it
    assert a[1] == pytest.approx(1070.0 / 2160 - 0.5)
    assert a[0] < a[1] < a[2]


def test_pinhole_fallback_uses_the_image_rows():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    cam = Camera(name="c", image_path="c", w2c=np.eye(4), K=K, width=640, height=480, time=0.0,
                 readout_s=0.02)
    assert sensor_of(cam).model == "PINHOLE"
    v = np.array([0.0, 240.0, 480.0])
    np.testing.assert_allclose(pixel_offsets(cam, np.full(3, 320.0), v), [-0.01, 0.0, 0.01])


torch = pytest.importorskip("torch")


def test_torch_matches_numpy_and_is_differentiable():
    x, y = grid(9)
    for raw in (FISHEYE, RawCamera.from_json({"model": "OPENCV", "K": RAW_K,
                                              "dist": [-0.28, 0.07, 0.001, -0.0007, -0.01],
                                              "w": 3840, "h": 2160})):
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        yt = torch.tensor(y, dtype=torch.float64, requires_grad=True)
        ft = row_fraction_normalized(xt, yt, raw)
        np.testing.assert_allclose(ft.detach().numpy(), row_fraction_normalized(x, y, raw), atol=1e-12)
        ft.sum().backward()
        assert torch.isfinite(yt.grad).all() and (yt.grad > 0).all()   # lower in the image, later
    # the fisheye at the exact optical axis has a finite gradient
    z = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    row_fraction_normalized(z, z, FISHEYE).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_rolling_shutter_offsets_follow_the_rows_and_the_motion():
    from cumuli_core.shutter import point_row_fraction, rolling_shutter_offsets

    raw = RawCamera.pinhole([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1]], 100, 100)
    w2c = torch.eye(4, dtype=torch.float64)
    pts = torch.tensor([[0.0, -0.25, 1.0], [0.0, 0.0, 1.0], [0.0, 0.25, 1.0], [0.0, 0.0, -1.0],
                        [0.0, 5.0, 1.0]], dtype=torch.float64)
    np.testing.assert_allclose(point_row_fraction(pts, w2c, raw).numpy(), [-0.25, 0.0, 0.25, 0.0, 0.5])

    # a point falling down the image (+y) at 1 unit/s; R = 0.1 s: the row is read
    # later as the point moves down, the fixed point of t = R * (y(t) / 1 * 100/100)
    R = 0.1
    vel = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    start = torch.tensor([[0.0, 0.1, 1.0]], dtype=torch.float64)
    means_at = lambda t: start + vel * torch.as_tensor(t, dtype=torch.float64).reshape(-1, 1)  # noqa: E731
    exact = R * 0.1 / (1 - R)       # t = R (0.1 + t)
    one = rolling_shutter_offsets(means_at, w2c, raw, 0.0, R, iterations=1)
    two = rolling_shutter_offsets(means_at, w2c, raw, 0.0, R, iterations=2)
    assert float(one) == pytest.approx(R * 0.1)
    assert abs(float(two) - exact) < 0.15 * abs(float(one) - exact)
    # a capture time that carries a gradient: the offsets do not, and do not break it
    tc = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    off = rolling_shutter_offsets(means_at, w2c, raw, tc, R)
    assert not off.requires_grad
    (means_at(tc + off)[:, 1].sum()).backward()
    assert float(tc.grad) == pytest.approx(1.0)

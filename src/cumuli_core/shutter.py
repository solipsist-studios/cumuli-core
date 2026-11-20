# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.shutter - rolling-shutter timing of a view.

A rolling-shutter sensor reads its lines one after another, top to bottom,
over a readout time R (seconds).  Light that fell on raw sensor row r of
H_raw rows was captured at

    t_pixel = time + time_offset + R * (r / H_raw - 0.5)

so a camera's `time_offset` is the moment of its middle line.  The training
and test images are undistorted pinhole frames, so a raw sensor row is a
curve in them: an undistorted pixel (u, v) goes to normalized coordinates
with the image's pinhole intrinsics, x = (u - cx) / fx, y = (v - cy) / fy,
is distorted with the raw camera model to raw pixel (u_raw, v_raw), and
r = v_raw.  Going through normalized coordinates makes the raw camera's
resolution independent of the dataset's downscale.

Raw models (`RawCamera.model`):

    OPENCV_FISHEYE  OpenCV fisheye (equidistant + k1..k4 polynomial):
                    theta = atan(r), theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8),
                    (x', y') = (theta_d / r) (x, y)
    OPENCV          Brown-Conrady: k1, k2, p1, p2[, k3[, k4, k5, k6]] (k4..k6 rational, as OpenCV)
    PINHOLE         no distortion (also what a camera with a readout but no raw
                    model uses: its image rows are its sensor rows)

then u_raw = fx x' + skew y' + cx, v_raw = fy y' + cy with the raw K.

The mapping functions take numpy arrays or torch tensors (differentiable:
only elementwise ops, no in-place writes to inputs).

Rendering a rolling-shutter view (`rolling_shutter_offsets`) gives every
Gaussian its own capture time: where its mean projects in the view decides
the raw row that saw it, and so the time to slice it at.  See that
function for why this method and what it costs.

Numpy; torch only when given tensors.
"""

from dataclasses import dataclass, field

import numpy as np

MODELS = {
    'OPENCV_FISHEYE': (4,),
    'OPENCV': (4, 5, 8),
    'PINHOLE': (0,),
}

#: Fixed-point passes of rolling_shutter_offsets: project at the middle-line
#: time, then once more at the time that row gives.  The second pass removes
#: the first-order error (the subject's own motion during the readout moves it
#: to another row); a third changes times by (v_rows * R / H)^2 of a readout,
#: under 1e-3 of R for anything that moves less than 3% of the frame height
#: per readout.
OFFSET_ITERATIONS = 2


@dataclass(frozen=True, eq=False)
class RawCamera:
    """The camera model of the raw (distorted) sensor image, at its full
    resolution.  K [3, 3] float64, dist a tuple of floats, width/height in
    raw pixels."""
    model: str
    K: np.ndarray = field(repr=False)
    dist: tuple
    width: int
    height: int

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f'raw_camera model {self.model!r}: expected one of {sorted(MODELS)}')
        if len(self.dist) not in MODELS[self.model]:
            raise ValueError(f'raw_camera model {self.model} takes {MODELS[self.model]} distortion '
                             f'coefficients, got {len(self.dist)}')
        if np.shape(self.K) != (3, 3):
            raise ValueError(f'raw_camera K must be 3x3, got shape {np.shape(self.K)}')
        if not self.height or self.height <= 0:
            raise ValueError(f'raw_camera h must be a positive pixel count, got {self.height!r}')

    @classmethod
    def from_json(cls, d) -> 'RawCamera':
        """From a transforms row's {"model", "K", "dist", "w", "h"}."""
        return cls(model=str(d['model']).upper(), K=np.asarray(d['K'], dtype=np.float64),
                   dist=tuple(float(v) for v in d.get('dist') or ()),
                   width=int(d['w']), height=int(d['h']))

    @classmethod
    def pinhole(cls, K, width, height) -> 'RawCamera':
        return cls(model='PINHOLE', K=np.asarray(K, dtype=np.float64), dist=(),
                   width=int(width or 0), height=int(height))

    def to_json(self) -> dict:
        return {'model': self.model, 'K': np.asarray(self.K).tolist(), 'dist': list(self.dist),
                'w': self.width, 'h': self.height}


def _xp(a):
    """The array module of a: torch for tensors, numpy otherwise."""
    if type(a).__module__.split('.')[0] == 'torch':
        import torch
        return torch
    return np


def distort_normalized(x, y, raw: RawCamera):
    """Raw-sensor pixel (u_raw, v_raw) of normalized undistorted coordinates
    (x, y) = (X / Z, Y / Z).  Elementwise; numpy arrays or torch tensors."""
    xp = _xp(x)
    d = raw.dist
    if raw.model == 'OPENCV_FISHEYE':
        k1, k2, k3, k4 = d
        r2 = x * x + y * y
        r = xp.sqrt(r2 + 1e-30)
        th = xp.arctan(r)
        th2 = th * th
        th_d = th * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
        # th_d / r -> 1 as r -> 0
        s = xp.where(r > 1e-8, th_d / xp.where(r > 1e-8, r, 1.0), 1.0 + 0.0 * r)
        xd, yd = x * s, y * s
    elif raw.model == 'OPENCV':
        k1, k2, p1, p2 = d[:4]
        k3 = d[4] if len(d) > 4 else 0.0
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        if len(d) == 8:
            k4, k5, k6 = d[5:]
            radial = radial / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xy = x * y
        xd = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
    else:
        xd, yd = x, y
    K = raw.K
    u = float(K[0, 0]) * xd + float(K[0, 1]) * yd + float(K[0, 2])
    v = float(K[1, 1]) * yd + float(K[1, 2])
    return u, v


def row_fraction_normalized(x, y, raw: RawCamera):
    """r / H_raw - 0.5 of the raw sensor row that saw normalized coordinates
    (x, y): -0.5 on the first line, 0 in the middle, +0.5 after the last."""
    _, v = distort_normalized(x, y, raw)
    return v / float(raw.height) - 0.5


def row_fraction(u, v, K, raw: RawCamera):
    """r / H_raw - 0.5 of the raw sensor row behind undistorted pixel
    coordinates (u, v) of an image with pinhole intrinsics K [3, 3] (OpenCV
    convention, the same one the dataset's fl/cx/cy use).  Not clamped."""
    K = np.asarray(K, dtype=np.float64)
    y = (v - float(K[1, 2])) / float(K[1, 1])
    x = (u - float(K[0, 2]) - float(K[0, 1]) * y) / float(K[0, 0])
    return row_fraction_normalized(x, y, raw)


def sensor_of(cam) -> RawCamera:
    """The raw camera that timed a view: its raw_camera, or, for a camera
    with a readout and no raw model, its own image read as the sensor."""
    raw = getattr(cam, 'raw_camera', None)
    if raw is not None:
        return raw
    return RawCamera.pinhole(cam.K, cam.width, cam.height)


def readout_of(cam) -> float:
    """The camera's readout time in seconds, 0 for a global shutter (and for
    cameras from an older cumuli_core)."""
    return float(getattr(cam, 'readout_s', 0.0) or 0.0)


def pixel_offsets(cam, u, v):
    """Seconds after the camera's capture_time at which undistorted pixel
    coordinates (u, v) were captured: R * (r / H_raw - 0.5).  numpy or torch."""
    R = readout_of(cam)
    return R * row_fraction(u, v, cam.K, sensor_of(cam))


def point_row_fraction(points, w2c, raw: RawCamera):
    """[N] raw row fraction of world points [N, 3] (torch) seen by a camera
    with world-to-camera w2c [4, 4] (OpenCV axes), clamped to [-0.5, 0.5];
    points on or behind the image plane get 0 (the middle line)."""
    import torch

    w2c = torch.as_tensor(w2c, dtype=points.dtype, device=points.device)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 1e-6
    zs = torch.where(front, z, torch.ones_like(z))
    frac = row_fraction_normalized(pc[:, 0] / zs, pc[:, 1] / zs, raw)
    frac = torch.nan_to_num(frac, nan=0.0).clamp(-0.5, 0.5)
    return torch.where(front, frac, torch.zeros_like(frac))


def rolling_shutter_offsets(means_at, w2c, raw: RawCamera, t_capture, readout_s,
                            iterations=OFFSET_ITERATIONS):
    """[N] seconds after `t_capture` at which a rolling-shutter view saw each
    Gaussian: readout_s * (raw row fraction where the Gaussian's mean
    projects).  Slice the Gaussians at `t_capture + offsets` (a per-Gaussian
    time; cumuli_core.spacetime.slice_at broadcasts it) and render once.

    means_at(t) returns the Gaussians' means [N, 3] at t (a float or an [N]
    tensor).  The row depends on where the Gaussian is when its row is read,
    so the time is found by fixed-point iteration: project at t_capture, take
    that row's time, project there, ... (`iterations` passes, 2 by default).
    Computed without gradient: the offsets are a sampling time, not a model
    output.  Gradients still reach anything inside t_capture (learned
    capture-time corrections) and the Gaussians' parameters through the
    slice the caller makes.

    Why per-Gaussian times.  The exact image is a different slice of the
    scene for every raw sensor row.  The alternatives:

      * per row band: render the view B times at B band times and keep each
        band's rows.  Exact as B grows, but B full renders (forward and
        backward) per view: B times the time, and B times the per-view
        render buffers, which training on a 32 GB card cannot afford.
      * per-Gaussian time (this): one render.  Each Gaussian is drawn at the
        time of the row its centre falls on, so the image is right wherever a
        pixel's colour comes from Gaussians whose footprint lies within a few
        rows of their centre - the error of a Gaussian is its speed times
        R x (its footprint height / H_raw), a fraction of the motion blur the
        real shutter shows anyway.  Large, fast Gaussians (bigger than the
        row band their motion crosses during the readout) are drawn at one
        time instead of being sheared; at R = 30 ms and a subject crossing
        the frame in a second, a Gaussian 5% of the frame tall is off by
        under 0.05 px of shear.
      * per-pixel times in the rasterizer: exact, but needs a custom gsplat
        kernel (and its backward).

    The cost is a projection of N means per pass and per view (no extra
    render, no extra image buffers), and the per-view Gaussians the batched
    renderer already holds when views have their own capture times.
    """
    import torch

    R = float(readout_s)
    if torch.is_tensor(t_capture):
        t_capture = t_capture.detach()
    with torch.no_grad():
        t = t_capture
        offsets = None
        for _ in range(max(1, int(iterations))):
            means = means_at(t)
            offsets = R * point_row_fraction(means.detach(), w2c, raw)
            t = t_capture + offsets
    return offsets

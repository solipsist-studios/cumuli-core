# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.cameras - cameras from a nerfstudio/blender transforms.json.

The file stores OpenGL camera-to-world matrices (x right, y up, camera looks
down -z).  gsplat and OpenCV want world-to-camera with y down and z forward.
load_transforms() flips the y and z axes of c2w and inverts it.

Intrinsics are either global (one fl_x/fl_y/cx/cy/w/h block at the top
level, n3v style) or per frame (custom rigs, and every dataset written by
build_flipbook_4dgs_dataset.py).  Per-frame values win when both exist.

Numpy only.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Camera:
    """One view: pose, intrinsics, image and timestamp.

    name       : basename of the entry's file_path, no directory.  Ground
                 truth for eval is looked up by this name.
    image_path : file_path resolved against the transforms file's directory.
                 It may lack an extension (the flipbook builder writes none);
                 Dataset4D.load_image() adds .png in that case.
    w2c        : [4, 4] float64 world-to-camera, OpenCV convention.
    K          : [3, 3] float64 pinhole intrinsics, already downscaled.
    width,
    height     : image size in pixels after downscale, or None when the
                 transforms file does not say.
    time       : seconds, from the entry's `time` (0.0 when absent).  The
                 nominal frame time: every camera of one synced instant has
                 the same value, so it is the key views are grouped by.
    label      : the entry's camera_label, or None.
    loss_weight: how much this view's photometric loss counts, from the
                 entry's `loss_weight` (1.0 when absent; 0 or more).  A
                 dataset marks views it trusts less, such as generated ones,
                 with a value below 1.  Trainers scale that view's colour
                 loss by it; the format says nothing about other terms.
    time_offset: seconds, from the entry's `time_offset` (0.0 when absent).
                 How much later than `time` this camera actually took the
                 picture (a sub-frame sync residual, PLUS convention:
                 capture = nominal + offset).
    readout_s  : seconds, from the entry's `readout_s` (0.0 when absent):
                 the rolling-shutter readout, first raw sensor line to last.
                 0 is a global shutter.  time_offset is the middle line's
                 time; see cumuli_core.shutter.
    raw_camera : cumuli_core.shutter.RawCamera from the entry's `raw_camera`
                 ({"model", "K", "dist", "w", "h"}, full raw resolution), or
                 None.  The distorted sensor the undistorted image came
                 from; it says which raw row (so which time) each pixel is.
    visit_weight: from the entry's `visit_weight` (1.0 when absent): how many
                 times as often training should visit this view's instant
                 (frame_motion.py gives fast frames more than 1).
    """
    name: str
    image_path: str
    w2c: np.ndarray
    K: np.ndarray
    width: Optional[int]
    height: Optional[int]
    time: float
    label: Optional[str] = None
    loss_weight: float = 1.0
    time_offset: float = 0.0
    readout_s: float = 0.0
    raw_camera: Optional[object] = field(default=None, repr=False, compare=False)
    visit_weight: float = 1.0
    extra: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def capture_time(self) -> float:
        """Seconds at which this view was actually captured: `time` plus the
        camera's sub-frame offset.  Render at this time to match the photo."""
        return self.time + self.time_offset

    @property
    def rolling_shutter(self) -> bool:
        """True when the view's rows were read at different times."""
        return self.readout_s != 0.0

    @property
    def c2w(self) -> np.ndarray:
        """[4, 4] camera-to-world, OpenCV convention."""
        return np.linalg.inv(self.w2c)

    @property
    def center(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return self.c2w[:3, 3]


def opengl_c2w_to_w2c(c2w_gl) -> np.ndarray:
    """OpenGL camera-to-world [4, 4] -> OpenCV world-to-camera [4, 4]."""
    c2w = np.array(c2w_gl, dtype=np.float64)
    c2w[:3, 1:3] *= -1.0
    return np.linalg.inv(c2w)


def _loss_weight(entry, path):
    """The entry's loss_weight as a float, 1.0 when absent.  A negative or
    non-finite weight would flip or poison the loss, so it is an error."""
    raw = entry.get('loss_weight', 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{path}: {entry.get("file_path")}: loss_weight '
                         f'{raw!r} is not a number') from None
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f'{path}: {entry.get("file_path")}: loss_weight must '
                         f'be finite and at least 0, got {raw!r}')
    return value


def load_transforms(path, downscale=1.0, every=1):
    """Read a transforms.json into a list of Camera, in file order.

    downscale divides the intrinsics and the image size (sizes round to the
    nearest pixel).  every keeps frames whose index is a multiple of it.
    """
    path = os.fspath(path)
    with open(path) as f:
        t = json.load(f)
    root = os.path.dirname(os.path.abspath(path))
    cams = []
    for i, f in enumerate(t['frames']):
        if i % every:
            continue
        # intrinsics are global (n3v-style) or per-frame (custom rigs)
        fx = f.get('fl_x', t.get('fl_x')) / downscale
        fy = f.get('fl_y', t.get('fl_y')) / downscale
        cx = f.get('cx', t.get('cx')) / downscale
        cy = f.get('cy', t.get('cy')) / downscale
        # Per-frame first, exactly like the intrinsics above. Reading these
        # only from the top level left them None for a dataset with
        # per-camera intrinsics, so a size check could not fire and a wrong
        # downscale silently rendered at the wrong scale.
        w = f.get('w', t.get('w'))
        h = f.get('h', t.get('h'))
        w = w and int(round(w / downscale))
        h = h and int(round(h / downscale))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        cams.append(Camera(
            name=os.path.basename(f['file_path']),
            image_path=os.path.normpath(os.path.join(root, f['file_path'])),
            w2c=opengl_c2w_to_w2c(f['transform_matrix']),
            K=K,
            width=w,
            height=h,
            time=float(f.get('time', 0.0)),
            label=f.get('camera_label'),
            loss_weight=_loss_weight(f, path),
            time_offset=float(f.get('time_offset', 0.0)),
            readout_s=float(f.get('readout_s') or 0.0),
            raw_camera=_raw_camera(f.get('raw_camera')),
            visit_weight=_visit_weight(f.get('visit_weight')),
        ))
    return cams


def _visit_weight(v):
    """A view's visit weight: 1.0 when absent; it must be a positive number."""
    if v is None:
        return 1.0
    w = float(v)
    if not w > 0:
        raise ValueError(f"visit_weight must be positive, got {v!r}")
    return w


def _raw_camera(d):
    if not d:
        return None
    from cumuli_core.shutter import RawCamera
    return RawCamera.from_json(d)


def group_by_time(cameras):
    """Map each distinct timestamp to its cameras, timestamps ascending and
    cameras in input order."""
    groups = {}
    for cam in cameras:
        groups.setdefault(cam.time, []).append(cam)
    return dict(sorted(groups.items()))

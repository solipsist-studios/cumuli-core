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
    time       : seconds, from the entry's `time` (0.0 when absent).
    label      : the entry's camera_label, or None.
    """
    name: str
    image_path: str
    w2c: np.ndarray
    K: np.ndarray
    width: Optional[int]
    height: Optional[int]
    time: float
    label: Optional[str] = None
    extra: dict = field(default_factory=dict, repr=False, compare=False)

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
        ))
    return cams


def group_by_time(cameras):
    """Map each distinct timestamp to its cameras, timestamps ascending and
    cameras in input order."""
    groups = {}
    for cam in cameras:
        groups.setdefault(cam.time, []).append(cam)
    return dict(sorted(groups.items()))

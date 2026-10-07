# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.dataset - reader for a dataset_4dgs/ training directory.

The layout is what cumuli's build_flipbook_4dgs_dataset.py writes:

    transforms_train.json    training views, per-frame intrinsics and time
    transforms_test.json     scored views (eval cameras, a held-out rig
                             camera, or one training camera standing in)
    realcams/cam<label>/frame_NNNNN.png        RGBA, subject mask in alpha
    evalcams/cam<label>/cam<label>_frame_NNNNN.png   only with --eval_root
    eval_gt_flat/<name>.png  scored views composited over black
    points3d.ply             visual-hull init points: x y z, uchar
                             red green blue, float time (seconds)

file_path entries carry no extension; load_image() tries the path as given,
then with .png, .jpg and .jpeg appended.

Numpy for everything; pillow for load_image().
"""

import os

import numpy as np

from cumuli_core.cameras import load_transforms

IMAGE_SUFFIXES = ('', '.png', '.jpg', '.jpeg')

_PLY_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


def read_ply_vertices(path):
    """Read the vertex element of a PLY into {property name: array}.

    Handles binary_little_endian and ascii files with scalar properties.
    Elements after the vertex element are ignored; list properties (faces)
    on the vertex element are not supported.  Returns (header_info, columns)
    where header_info has 'format', 'count' and 'comments'.
    """
    path = os.fspath(path)
    with open(path, 'rb') as fp:
        header_lines = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f'{path}: truncated PLY header')
            text = line.decode('ascii', 'replace').strip()
            header_lines.append(text)
            if text == 'end_header':
                break
        if header_lines[0] != 'ply':
            raise ValueError(f'{path}: not a PLY file')

        fmt = None
        comments = []
        elements = []                      # [name, count, [(prop, dtype)]]
        for text in header_lines[1:-1]:
            parts = text.split()
            if not parts:
                continue
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'comment':
                comments.append(text[len('comment'):].strip())
            elif parts[0] == 'element':
                elements.append([parts[1], int(parts[2]), []])
            elif parts[0] == 'property':
                if not elements:
                    raise ValueError(f'{path}: property before any element')
                if parts[1] == 'list':
                    elements[-1][2].append((parts[-1], None))
                else:
                    if parts[1] not in _PLY_TYPES:
                        raise ValueError(f'{path}: unknown PLY type {parts[1]!r}')
                    elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]]))

        if fmt not in ('binary_little_endian', 'ascii'):
            raise ValueError(f'{path}: unsupported PLY format {fmt!r}')
        if not elements or elements[0][0] != 'vertex':
            raise ValueError(f'{path}: first element must be "vertex"')
        _, n, props = elements[0]
        if any(dt is None for _, dt in props):
            raise ValueError(f'{path}: list properties on vertices are not supported')

        if fmt == 'binary_little_endian':
            dtype = np.dtype([(name, '<' + dt) for name, dt in props])
            data = np.fromfile(fp, dtype=dtype, count=n)
            if len(data) != n:
                raise ValueError(f'{path}: truncated body, expected {n} vertices, '
                                 f'got {len(data)}')
            columns = {name: np.ascontiguousarray(data[name]) for name, _ in props}
        else:
            rows = []
            for _ in range(n):
                line = fp.readline()
                if not line:
                    raise ValueError(f'{path}: truncated body, expected {n} vertices')
                rows.append(line.split()[:len(props)])
            table = np.array(rows, dtype=np.float64).reshape(n, len(props))
            columns = {name: table[:, i].astype(dt)
                       for i, (name, dt) in enumerate(props)}
    return {'format': fmt, 'count': n, 'comments': comments}, columns


class Dataset4D:
    """A dataset_4dgs/ directory: cameras, RGBA images, init points.

    train_cameras, test_cameras : lists of Camera from the transforms files
                                  (test is empty if transforms_test.json is
                                  absent).
    time_min, time_max          : span of every camera timestamp, seconds.
    eval_gt_dir                 : eval_gt_flat/ path, or None if absent.
    """

    def __init__(self, root, downscale=1.0):
        self.root = os.path.abspath(os.fspath(root))
        train = os.path.join(self.root, 'transforms_train.json')
        if not os.path.exists(train):
            raise FileNotFoundError(f'{train}: not a dataset_4dgs directory')
        self.train_cameras = load_transforms(train, downscale=downscale)
        test = os.path.join(self.root, 'transforms_test.json')
        self.test_cameras = (load_transforms(test, downscale=downscale)
                             if os.path.exists(test) else [])
        times = [c.time for c in self.train_cameras + self.test_cameras]
        self.time_min = float(min(times)) if times else 0.0
        self.time_max = float(max(times)) if times else 0.0
        gt = os.path.join(self.root, 'eval_gt_flat')
        self.eval_gt_dir = gt if os.path.isdir(gt) else None

    @property
    def times(self):
        """Sorted distinct training timestamps."""
        return sorted({c.time for c in self.train_cameras})

    def image_file(self, camera):
        """The on-disk image for `camera`, trying the known extensions."""
        for suffix in IMAGE_SUFFIXES:
            path = camera.image_path + suffix
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f'no image for {camera.image_path} '
                                f'(tried suffixes {IMAGE_SUFFIXES})')

    def load_image(self, camera):
        """RGBA float32 [H, W, 4] in [0, 1].  Alpha is 1 when the file has none.

        Colour is NOT premultiplied: the files store straight RGB with the
        mask in alpha.  Composite with rgb * a + bg * (1 - a) to compare
        against a render over background bg.
        """
        from PIL import Image
        with Image.open(self.image_file(camera)) as im:
            arr = np.asarray(im.convert('RGBA'), dtype=np.float32) / 255.0
        if camera.width and camera.height and arr.shape[:2] != (camera.height, camera.width):
            raise ValueError(f'{camera.name}: image is {arr.shape[1]}x{arr.shape[0]} but '
                             f'the transforms say {camera.width}x{camera.height}')
        return arr

    def gt_path(self, camera):
        """eval_gt_flat/<name>.png for a test camera, or None."""
        if self.eval_gt_dir is None:
            return None
        path = os.path.join(self.eval_gt_dir, camera.name + '.png')
        return path if os.path.isfile(path) else None

    def load_points(self, path=None):
        """points3d.ply as a dict of arrays.

        Always: xyz [N, 3] float32.  When present: rgb [N, 3] float32 in
        [0, 1] (from uchar red/green/blue, or float in [0, 1]), time [N]
        float32, normals [N, 3].  Every other vertex property is passed
        through under its own name.
        """
        path = path or os.path.join(self.root, 'points3d.ply')
        _, cols = read_ply_vertices(path)
        out = {'xyz': np.stack([cols.pop(c) for c in ('x', 'y', 'z')],
                               axis=1).astype(np.float32)}
        if all(c in cols for c in ('red', 'green', 'blue')):
            rgb = np.stack([cols.pop(c) for c in ('red', 'green', 'blue')], axis=1)
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(np.float32) / 255.0
            out['rgb'] = rgb.astype(np.float32)
        if 'time' in cols:
            out['time'] = cols.pop('time').astype(np.float32)
        if all(c in cols for c in ('nx', 'ny', 'nz')):
            out['normals'] = np.stack([cols.pop(c) for c in ('nx', 'ny', 'nz')],
                                      axis=1).astype(np.float32)
        out.update(cols)
        return out

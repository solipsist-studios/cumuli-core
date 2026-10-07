# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.sogst - decode a .sogst archive back into per-splat fields.

This mirrors the engine decoder.  The packer's own verifier covers the same
math for scalar attributes; this adds quats and shN, so it is the only
complete inverse of the encoder.  The result has the same layout as
cumuli_core.ply.read_sogst_ply(), so either source feeds the renderer.

Numpy plus pillow (imported on first use, for the webp textures).
"""

import io
import json
import zipfile

import numpy as np

from cumuli_core.ply import read_sogst_ply
from cumuli_core.sh import SOGST_SHN_COEFFS


def decode_webp(blob: bytes) -> np.ndarray:
    """Decode one lossless webp texture to [H, W, 4] uint8 RGBA."""
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(blob)).convert('RGBA'))


def decode_sogst_fields(v3_path):
    """Decode a .sogst archive into the field dict the packer consumes,
    plus a (time_min, time_max, fps) header tuple."""
    zf = zipfile.ZipFile(v3_path)
    meta = json.loads(zf.read('meta.json'))
    n = meta['count']

    if meta.get('streams'):
        groups = []
        if meta['streams']['persistent']:
            groups.append((meta['streams']['persistent'], meta['segments']['persistent']))
        for prefix, seg in zip(meta['streams']['segments'], meta['segments']['list']):
            if prefix:
                groups.append((prefix, seg['range']))
        names = {name.split('/', 1)[1] for name in zf.namelist() if '/' in name}
        tex = {name: np.zeros((n, 4), dtype=np.uint8) for name in names}
        for prefix, (a, b) in groups:
            for name in names:
                tex[name][a:b] = decode_webp(zf.read(f'{prefix}/{name}')).reshape(-1, 4)[:b - a]
        cent_raw = zf.read('shN_centroids.webp') if 'shN_centroids.webp' in zf.namelist() else None
    else:
        tex = {name: decode_webp(zf.read(name)).reshape(-1, 4)[:n]
               for name in zf.namelist() if name.endswith('.webp') and name != 'shN_centroids.webp'}
        cent_raw = zf.read('shN_centroids.webp') if 'shN_centroids.webp' in zf.namelist() else None

    def unsplit16(l, u, mins, maxs):
        q = (u.astype(np.float64) * 256 + l) / 65535.0
        t = np.asarray(mins)[None, :] + q * (np.asarray(maxs)[None, :] - np.asarray(mins)[None, :])
        return np.sign(t) * (np.exp(np.abs(t)) - 1.0)

    fields = {}
    xyz = unsplit16(tex['means_l.webp'][:, :3], tex['means_u.webp'][:, :3],
                    meta['means']['mins'], meta['means']['maxs'])
    fields['x'], fields['y'], fields['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # smallest-three quats: byte planes are the three kept components in
    # [-1/sqrt(2), 1/sqrt(2)], alpha = 252 + index of the dropped (largest)
    # component in wxyz order
    qb = (tex['quats.webp'][:, :3].astype(np.float64) / 255.0 - 0.5) * np.sqrt(2.0)
    mode = tex['quats.webp'][:, 3].astype(np.int64) - 252
    d = np.sqrt(np.clip(1.0 - (qb * qb).sum(axis=1), 0.0, None))
    quat = np.empty((n, 4))
    # engine mapping (GSplatSogIterator): mode 0 -> (a,b,c,d) as x,y,z,w ...
    # expressed in w-first storage below (rot_0 = w)
    a, b, c = qb[:, 0], qb[:, 1], qb[:, 2]
    for m, (w_, x_, y_, z_) in enumerate((
            (d, a, b, c), (a, d, b, c), (a, b, d, c), (a, b, c, d))):
        sel = mode == m
        quat[sel, 0] = w_[sel]
        quat[sel, 1] = x_[sel]
        quat[sel, 2] = y_[sel]
        quat[sel, 3] = z_[sel]
    for i in range(4):
        fields[f'rot_{i}'] = quat[:, i]

    scales_cb = np.asarray(meta['scales']['codebook'])
    for c_ in range(3):
        fields[f'scale_{c_}'] = scales_cb[tex['scales.webp'][:, c_]]

    sh0_cb = np.asarray(meta['sh0']['codebook'])
    for c_ in range(3):
        fields[f'f_dc_{c_}'] = sh0_cb[tex['sh0.webp'][:, c_]]
    alpha = tex['sh0.webp'][:, 3].astype(np.float64) / 255.0
    alpha = np.clip(alpha, 1e-5, 1.0 - 1e-5)
    fields['opacity'] = np.log(alpha / (1.0 - alpha))

    vel = unsplit16(tex['motion_l.webp'][:, :3], tex['motion_u.webp'][:, :3],
                    meta['motion']['mins'], meta['motion']['maxs'])
    fields['vx'], fields['vy'], fields['vz'] = vel[:, 0], vel[:, 1], vel[:, 2]

    if meta.get('accel'):
        acc = unsplit16(tex['accel_l.webp'][:, :3], tex['accel_u.webp'][:, :3],
                        meta['accel']['mins'], meta['accel']['maxs'])
        fields['ax'], fields['ay'], fields['az'] = acc[:, 0], acc[:, 1], acc[:, 2]

    fields['t_center'] = np.asarray(meta['trbf']['center']['codebook'])[tex['trbf.webp'][:, 0]]
    fields['t_sigma'] = np.asarray(meta['trbf']['sigma']['codebook'])[tex['trbf.webp'][:, 1]]

    if meta.get('shN') and cent_raw is not None:
        cent = decode_webp(cent_raw)                       # [H, W, 4] RGBA bytes
        h, w = cent.shape[0], cent.shape[1]
        flat = cent.reshape(h * w, 4)
        labels = (tex['shN_labels.webp'][:, 0].astype(np.int64) +
                  (tex['shN_labels.webp'][:, 1].astype(np.int64) << 8))
        codebook = np.asarray(meta['shN']['codebook'])
        coeffs = SOGST_SHN_COEFFS[meta['shN']['bands']]
        # engine layout: palette entry n occupies texels
        # [(n % 64) * coeffs, (n % 64 + 1) * coeffs) on row n // 64.
        # sh[j*coeffs + k] = codebook[centroid_bytes[(u + k)*4 + j + v*W*4]]
        u = (labels % 64) * coeffs
        v = labels // 64
        base = v * w + u                                    # texel index of coeff 0
        f_rest = np.zeros((n, 3 * coeffs), dtype=np.float64)
        for k in range(coeffs):
            texel = flat[base + k]                          # [N, 4] bytes
            for j in range(3):
                f_rest[:, j * coeffs + k] = codebook[texel[:, j]]
        fields['f_rest'] = f_rest

    zf.close()
    time = meta.get('time', {})
    header = {'time_min': time.get('min', 0.0), 'time_max': time.get('max', 0.0),
              'fps': time.get('fps', 30.0), 'count': n}
    return header, fields


def load_model(path):
    """Load a .sogst archive, or a 4D interchange PLY (unquantized, so
    scoring one against its packed archive isolates quantization cost)."""
    if zipfile.is_zipfile(path):
        return decode_sogst_fields(path)
    header, fields = read_sogst_ply(path)
    return {'time_min': header['time_min'], 'time_max': header['time_max'],
            'fps': header['fps'], 'count': header['count']}, fields

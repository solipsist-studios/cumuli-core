# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.trigsbin - read an Open4DGS ``.trigsbin`` runtime file.

The layout is the one written by Open4DGS-Core's ``deliver/make-trigs-timeline.py``
(MIT, https://github.com/ASU-NEM-Labs/open4dgs-core, v0.1.0, runtime format version 1):

    [64-byte header][UTF-8 JSON manifest][zero padding][aligned raw tensor payload]

    header = "<8sIIQQQQQQ": magic b"TRIGSBIN", version, header_bytes, manifest_offset,
             manifest_length, payload_offset, payload_length, alignment, reserved

Each tensor is described by the manifest's ``tensorIndex`` (and by the ``tensors`` maps of
each clip's ``dynamic`` group and of the top-level ``static`` group); its ``byteOffset`` is
relative to ``payload_offset``.  Quaternions are stored xyzw, scales as natural logs and
opacity as a logit (``semantics``).  A clip's times are relative to its training start when
the file is "clip-local" (the packager's default).

``read`` returns the parsed file; ``TrigsModel`` slices it at a global time, picking the clips
that cover that time and fading overlaps as the manifest says.  Needs numpy; the model needs
torch.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b'TRIGSBIN'
HEADER = struct.Struct('<8sIIQQQQQQ')
MOTION_MODEL = 'trigs.se3_bezier_v1'

_DTYPES = {'float32': '<f4', 'float16': '<f2', 'uint8': 'u1', 'int32': '<i4', 'uint32': '<u4'}

DYNAMIC_KEYS = ('means', 'scales', 'quats', 'opacities', 'sh0', 'shN', 'times', 'log_durations', 'anchors',
                'zeta_base_r', 'zeta_base_t', 'zeta_c0_r', 'zeta_c0_t', 'zeta_c1_r', 'zeta_c1_t',
                'zeta_c2_r', 'zeta_c2_t')


class TrigsbinError(ValueError):
    pass


@dataclass
class Clip:
    id: str
    #: Global frame range at full opacity, [start, end).
    core: tuple[int, int]
    #: Global frame range the clip is drawn in at all (core plus its fades), [start, end).
    active: tuple[int, int]
    fade_in_frames: int
    fade_out_frames: int
    fade_curve: str
    #: Seconds to add to this clip's tensor times to get global time.
    time_origin_s: float
    #: Dynamic tensors by name, float32, in the file's own conventions (xyzw quats).
    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.tensors['means'].shape[0]) if 'means' in self.tensors else 0


@dataclass
class TrigsFile:
    path: Path
    manifest: dict
    clips: list[Clip]
    #: Splats that never move, shared by every clip (empty when there are none).
    static: dict[str, np.ndarray]

    @property
    def fps(self) -> float:
        return float(self.manifest['timeline']['frameRate'])

    @property
    def total_frames(self) -> int:
        return int(self.manifest['timeline']['totalFrames'])

    @property
    def duration(self) -> float:
        return float(self.manifest['timeline'].get('durationSeconds') or self.total_frames / self.fps)

    @property
    def motion_model(self) -> str:
        return self.manifest.get('semantics', {}).get('dynamicMotionModel', '')

    @property
    def count(self) -> int:
        return sum(c.count for c in self.clips) + (int(self.static['means'].shape[0]) if 'means' in self.static else 0)


def read_header(f) -> dict:
    raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise TrigsbinError('the file is shorter than a .trigsbin header')
    magic, version, header_bytes, m_off, m_len, p_off, p_len, align, _ = HEADER.unpack(raw)
    if magic != MAGIC:
        raise TrigsbinError(f'not a .trigsbin file (magic {magic!r})')
    if version != 1:
        raise TrigsbinError(f'unsupported .trigsbin version {version}')
    return {'version': version, 'header_bytes': header_bytes, 'manifest_offset': m_off, 'manifest_length': m_len,
            'payload_offset': p_off, 'payload_length': p_len, 'alignment': align}


def read_manifest(path) -> tuple[dict, dict]:
    """(header, manifest) without touching the payload."""
    with open(path, 'rb') as f:
        header = read_header(f)
        f.seek(header['manifest_offset'])
        manifest = json.loads(f.read(header['manifest_length']).decode('utf-8'))
    return header, manifest


def _tensor(f, header: dict, spec: dict) -> np.ndarray:
    dtype = _DTYPES.get(spec['dtype'])
    if dtype is None:
        raise TrigsbinError(f"unknown tensor dtype {spec['dtype']!r}")
    shape = tuple(int(v) for v in spec['shape'])
    count = int(np.prod(shape)) if shape else 1
    f.seek(header['payload_offset'] + int(spec['byteOffset']))
    data = np.fromfile(f, dtype=dtype, count=count)
    if data.size != count:
        raise TrigsbinError(f"tensor {spec.get('path')} is cut short")
    return data.reshape(shape)


def _group(f, header: dict, tensors: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, spec in tensors.items():
        data = _tensor(f, header, spec)
        out[name] = data.astype(np.float32, copy=False) if spec['dtype'].startswith('float') else data
    return out


def read(path) -> TrigsFile:
    """Parse a .trigsbin: every clip's dynamic tensors (as float32) and the shared static group."""
    path = Path(path)
    header, manifest = read_manifest(path)
    if manifest.get('format') != 'trigs.runtime':
        raise TrigsbinError(f"the manifest says format {manifest.get('format')!r}, not 'trigs.runtime'")
    if manifest.get('semantics', {}).get('dynamicMotionModel') != MOTION_MODEL:
        raise TrigsbinError(f"unsupported motion model {manifest.get('semantics', {}).get('dynamicMotionModel')!r}")
    fps = float(manifest['timeline']['frameRate'])
    domain = manifest.get('semantics', {}).get('dynamicTimesDomain', 'clip-local')
    clips = []
    with open(path, 'rb') as f:
        for raw in manifest.get('clips', []):
            glob, render, training = raw['global'], raw['render'], raw['training']
            rebased = (raw.get('source', {}).get('temporalSelection') or {}).get('timesRebasedBySeconds')
            if rebased is not None:
                origin = float(rebased)
            else:
                origin = 0.0 if domain == 'global' else float(training['globalFrameStart']) / fps
            clips.append(Clip(
                id=raw['id'],
                core=(int(glob['coreFrameStart']), int(glob['coreFrameEndExclusive'])),
                active=(int(glob['activeFrameStart']), int(glob['activeFrameEndExclusive'])),
                fade_in_frames=int(render.get('fadeInFrames', 0)), fade_out_frames=int(render.get('fadeOutFrames', 0)),
                fade_curve=str(render.get('fadeCurve', 'smoothstep')), time_origin_s=origin,
                tensors=_group(f, header, raw['dynamic']['tensors'])))
        static = _group(f, header, (manifest.get('static') or {}).get('tensors') or {})
    return TrigsFile(path=path, manifest=manifest, clips=clips, static=static)


def _fade(x, curve: str):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x) if curve == 'smoothstep' else x


def clip_weight(clip: Clip, frame: float) -> float:
    """Opacity weight of a clip at a (fractional) global frame: 1 inside its core, a fade across the frames
    before and after it, 0 outside.  A clip with no fades is a hard cut at its core."""
    lo, hi = clip.core
    if lo <= frame < hi or (frame == hi and hi == clip.active[1]):
        return 1.0
    if frame < lo and clip.fade_in_frames > 0:
        return float(_fade((frame - (lo - clip.fade_in_frames)) / clip.fade_in_frames, clip.fade_curve))
    if frame >= hi and clip.fade_out_frames > 0:
        return float(_fade(((hi + clip.fade_out_frames) - frame) / clip.fade_out_frames, clip.fade_curve))
    return 0.0


class TrigsModel:
    """A .trigsbin on a torch device, sliceable at a global time like ``cumuli_core.spacetime``'s tensors.

    ``slice(t)`` returns the splats drawn at global time t as (means [N,3], quats_wxyz [N,4], opacities [N],
    sh [N,K,3], scales [N,3]); clips that overlap are both drawn, each weighted by its fade.  Splats of the
    shared static group, if any, are always drawn.  Per-view rolling-shutter slicing needs one clip (the common
    case: one trainer window); ``slice`` raises for a per-splat time on several clips."""

    def __init__(self, tf: TrigsFile, device='cpu') -> None:
        import torch

        from cumuli_core.trigs import TrigsTensors

        self.file = tf
        self.device = torch.device(device)
        self.fps = tf.fps
        self.time_min = 0.0
        self.time_max = tf.duration
        self._clips = [(c, self._tensors(c.tensors, TrigsTensors, torch)) for c in tf.clips if c.count]
        self.static = self._static(tf.static, torch) if 'means' in tf.static and tf.static['means'].shape[0] else None
        self.sh_degree = max((p.sh_degree for _, p in self._clips), default=0)

    # -- construction -----------------------------------------------------------------------------
    def _t(self, a, torch):
        return torch.tensor(np.ascontiguousarray(a, dtype=np.float32), dtype=torch.float32, device=self.device)

    def _tensors(self, d: dict, cls, torch):
        sh0 = self._t(d['sh0'], torch).reshape(-1, 1, 3)
        shn = self._t(d['shN'], torch).reshape(sh0.shape[0], -1, 3) if 'shN' in d else sh0[:, :0]
        quats = self._t(d['quats'], torch)
        wxyz = torch.cat([quats[:, 3:4], quats[:, :3]], dim=1)   # the file stores xyzw
        return cls(means=self._t(d['means'], torch), quats_wxyz=wxyz, log_scales=self._t(d['scales'], torch),
                   opacity_logit=self._t(d['opacities'], torch).reshape(-1), sh=torch.cat([sh0, shn], dim=1),
                   times=self._t(d['times'], torch).reshape(-1), log_sigma=self._t(d['log_durations'], torch).reshape(-1),
                   anchors=self._t(d['anchors'], torch),
                   **{k: self._t(d[k], torch) for k in ('zeta_base_r', 'zeta_base_t', 'zeta_c0_r', 'zeta_c0_t',
                                                        'zeta_c1_r', 'zeta_c1_t', 'zeta_c2_r', 'zeta_c2_t')})

    def _static(self, d: dict, torch):
        sh0 = self._t(d['sh0'], torch).reshape(-1, 1, 3)
        shn = self._t(d['shN'], torch).reshape(sh0.shape[0], -1, 3) if 'shN' in d else sh0[:, :0]
        quats = self._t(d['quats'], torch)
        return {'means': self._t(d['means'], torch), 'quats': torch.cat([quats[:, 3:4], quats[:, :3]], dim=1),
                'scales': self._t(d['scales'], torch), 'opacity': torch.sigmoid(self._t(d['opacities'], torch).reshape(-1)),
                'sh': torch.cat([sh0, shn], dim=1)}

    # -- slicing ----------------------------------------------------------------------------------
    def slice(self, t):
        import torch

        from cumuli_core.trigs import slice_at

        per_splat = isinstance(t, torch.Tensor) and t.dim() > 0
        if per_splat and len(self._clips) != 1:
            raise TrigsbinError('a per-splat time (rolling shutter) needs a file with one clip')
        parts = []
        frame = (float(t) if not per_splat else float(t.mean())) * self.fps
        for clip, params in self._clips:
            w = clip_weight(clip, frame)
            if w <= 0.0:
                continue
            local = t - clip.time_origin_s
            means, quats, opac = slice_at(params, local)
            parts.append((means, quats, opac * w, params.sh, params.scales))
        if self.static is not None:
            s = self.static
            parts.append((s['means'], s['quats'], s['opacity'], s['sh'], torch.exp(s['scales'])))
        if not parts:
            z = torch.zeros
            k = max(1, (self.sh_degree + 1) ** 2)
            return (z(0, 3, device=self.device), z(0, 4, device=self.device), z(0, device=self.device),
                    z(0, k, 3, device=self.device), z(0, 3, device=self.device))
        if len(parts) == 1:
            return parts[0]
        k = max(p[3].shape[1] for p in parts)
        sh = [torch.nn.functional.pad(p[3], (0, 0, 0, k - p[3].shape[1])) for p in parts]
        return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]),
                torch.cat(sh), torch.cat([p[4] for p in parts]))


def load_model(path, device='cpu') -> TrigsModel:
    return TrigsModel(read(path), device)

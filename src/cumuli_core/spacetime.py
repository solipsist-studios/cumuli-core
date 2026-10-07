# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.spacetime - per-splat spacetime tensors and the temporal slice.

A 4D splat is a 3D gaussian whose mean moves and whose opacity is windowed in
time.  At time t, with dt = t - t_center:

    mean(t)  = xyz + velocity * dt (+ accel * dt^2)
    alpha(t) = sigmoid(opacity_logit) * exp(-0.5 * (dt / t_sigma)^2)

accel is the raw dt^2 coefficient, not a/2 (docs/sogst-format.md section 7).
Everything else (rotation, scale, colour) is static.

SpacetimeTensors holds the parameters in the spaces the interchange PLY
stores them in: wxyz quaternions, natural-log scales, logit opacity, raw SH
coefficients, t_sigma as a standard deviation in seconds.

Needs torch.
"""

from dataclasses import dataclass, fields as dc_fields
from typing import Optional

import numpy as np
import torch

from cumuli_core.ply import PLY_ACCEL_COLUMNS, PLY_BASE_COLUMNS
from cumuli_core.sh import shn_bands_for_width

# Floor applied to |t_sigma| on load, as eval_render.py always has.  A valid
# PLY has t_sigma > 0 already, so this only guards a hand-built field dict.
T_SIGMA_FLOOR = 1e-6


@dataclass
class SpacetimeTensors:
    """Per-splat parameters, all float32 on one device.

    xyz           [N, 3]    position at t_center
    velocity      [N, 3]    scene units per second
    accel         [N, 3]    raw dt^2 coefficient, or None for linear motion
    quats_wxyz    [N, 4]    rotation, w first, not necessarily normalized
    log_scales    [N, 3]    natural-log scale
    opacity_logit [N]       peak opacity (at t_center), logit space
    t_center      [N]       seconds, absolute clip time
    t_sigma       [N]       temporal standard deviation, seconds, > 0
    sh            [N, K, 3] SH coefficients, K = (degree + 1)^2, DC at k = 0
    """
    xyz: torch.Tensor
    velocity: torch.Tensor
    accel: Optional[torch.Tensor]
    quats_wxyz: torch.Tensor
    log_scales: torch.Tensor
    opacity_logit: torch.Tensor
    t_center: torch.Tensor
    t_sigma: torch.Tensor
    sh: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def sh_degree(self) -> int:
        """SH degree implied by sh.shape[1]: 1, 4, 9, 16 -> 0, 1, 2, 3."""
        k = int(self.sh.shape[1])
        degree = int(round(k ** 0.5)) - 1
        if (degree + 1) ** 2 != k:
            raise ValueError(f'sh has {k} coefficients per channel, not a square')
        return degree

    @property
    def scales(self) -> torch.Tensor:
        """Linear scales, exp(log_scales)."""
        return torch.exp(self.log_scales)

    def slice(self, t):
        """(means [N, 3], opacities [N]) at time t.  See slice_at()."""
        return slice_at(self.xyz, self.velocity, self.opacity_logit,
                        self.t_center, self.t_sigma, t, accel=self.accel)

    def to(self, device) -> 'SpacetimeTensors':
        moved = {f.name: (None if getattr(self, f.name) is None
                          else getattr(self, f.name).to(device))
                 for f in dc_fields(self)}
        return SpacetimeTensors(**moved)


def slice_at(xyz, velocity, opacity_logit, t_center, t_sigma, t, accel=None):
    """Evaluate the temporal model at time t.

        dt      = t - t_center
        means   = xyz + velocity * dt (+ accel * dt^2)
        opacity = sigmoid(opacity_logit) * exp(-0.5 * (dt / t_sigma)^2)

    t is a Python float or a 0-d tensor.  Differentiable with respect to
    every tensor input, t included when it is a tensor: no in-place ops on
    the inputs and no host syncs.  Returns (means [N, 3], opacities [N]).
    """
    dt = t - t_center
    dt3 = dt.unsqueeze(-1)
    means = xyz + velocity * dt3
    if accel is not None:
        means = means + accel * (dt3 * dt3)
    opacities = torch.sigmoid(opacity_logit) * torch.exp(-0.5 * (dt / t_sigma) ** 2)
    return means, opacities


def from_fields(fields, device='cpu') -> SpacetimeTensors:
    """Build SpacetimeTensors from a PLY-layout field dict, as returned by
    cumuli_core.ply.read_sogst_ply() or cumuli_core.sogst.load_model().

    f_rest [N, 3 * (K - 1)] is channel-major (column j * (K - 1) + k is
    channel j, coefficient k), so sh[:, 1:, :] is f_rest reshaped to
    [N, 3, K - 1] and transposed.  This is exactly what eval_render.py does.
    """
    device = torch.device(device)

    def to(a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

    def stack(names):
        return to(np.stack([np.asarray(fields[c]) for c in names], axis=1))

    n = len(np.asarray(fields['x']))
    accel = stack(PLY_ACCEL_COLUMNS) if 'ax' in fields else None
    sh_degree = shn_bands_for_width(fields['f_rest'].shape[1]) if 'f_rest' in fields else 0
    n_sh = (sh_degree + 1) ** 2
    sh = torch.zeros((n, n_sh, 3), dtype=torch.float32, device=device)
    sh[:, 0, :] = stack(['f_dc_0', 'f_dc_1', 'f_dc_2'])
    if sh_degree:
        f_rest = to(fields['f_rest'])
        sh[:, 1:, :] = f_rest.reshape(n, 3, n_sh - 1).permute(0, 2, 1)

    return SpacetimeTensors(
        xyz=stack(['x', 'y', 'z']),
        velocity=stack(['vx', 'vy', 'vz']),
        accel=accel,
        quats_wxyz=stack([f'rot_{i}' for i in range(4)]),
        log_scales=stack([f'scale_{i}' for i in range(3)]),
        opacity_logit=to(fields['opacity']),
        t_center=to(fields['t_center']),
        t_sigma=to(np.maximum(np.abs(np.asarray(fields['t_sigma'])), T_SIGMA_FLOOR)),
        sh=sh,
    )


def to_fields(st: SpacetimeTensors) -> dict:
    """Inverse of from_fields(): a PLY-layout dict of float32 numpy arrays
    that write_sogst_ply() accepts.  f_rest is present only when the SH
    degree is above 0, and ax/ay/az only when accel is set."""
    def np_(x):
        return x.detach().to('cpu', torch.float32).numpy()

    out = {}
    for prefix, tensor in (('xyz', st.xyz), ('v', st.velocity)):
        arr = np_(tensor)
        names = ['x', 'y', 'z'] if prefix == 'xyz' else ['vx', 'vy', 'vz']
        for i, name in enumerate(names):
            out[name] = arr[:, i]
    q = np_(st.quats_wxyz)
    for i in range(4):
        out[f'rot_{i}'] = q[:, i]
    s = np_(st.log_scales)
    for i in range(3):
        out[f'scale_{i}'] = s[:, i]
    out['opacity'] = np_(st.opacity_logit)
    sh = np_(st.sh)
    for i in range(3):
        out[f'f_dc_{i}'] = sh[:, 0, i]
    out['t_center'] = np_(st.t_center)
    out['t_sigma'] = np_(st.t_sigma)
    n, k, _ = sh.shape
    if k > 1:
        out['f_rest'] = np.ascontiguousarray(
            sh[:, 1:, :].transpose(0, 2, 1).reshape(n, 3 * (k - 1)))
    if st.accel is not None:
        a = np_(st.accel)
        for i, name in enumerate(PLY_ACCEL_COLUMNS):
            out[name] = a[:, i]
    # Base columns first, in spec order, for readability of the dict.
    ordered = {name: out[name] for name in PLY_BASE_COLUMNS}
    ordered.update({k_: v for k_, v in out.items() if k_ not in ordered})
    return ordered

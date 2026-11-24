# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.trigs - the TRiGS motion model and its temporal slice.

A TRiGS splat (Open4DGS "trigs.se3_bezier_v1") is a 3D gaussian with a temporal
window and a motion that is a rotation and translation about an anchor, whose
twist varies in time along a quadratic Bezier curve.  This module is the one
reference evaluation: the trainer, the evaluator and the viewer's parity test
all slice with it.

Written from the TRiGS paper (arXiv 2604.00538, Sec. 3, Eqs. 1 and 4-10) and
from the tensor names of the MIT Open4DGS-Core packager
(deliver/make-trigs-timeline.py).  No trainer code was read.  Where the paper
is silent the choice is ours and is marked "ours".

For one splat, with t_c its time centre (``times``), s its temporal sigma
(``exp(log_durations)``) and dt = t - t_c:

    gamma(t)  = exp(-0.5 (dt / s)^2)                   (Eq. 1)
    w(t)      = max((gamma - 0.05) / 0.95, 0)          the weight the file format applies (see below)
    alpha(t)  = sigmoid(opacity) * w
    tau       = clamp((dt + 2 s) / (4 s), 0, 1)        (Eqs. 5-6)
    zeta(t)   = zeta_base + (1-tau)^2 c0 + 2 (1-tau) tau c1 + tau^2 c2        (Eqs. 7-8)
    phi       = omega(t) dt      upsilon = nu(t) dt    (zeta = (omega, nu) in se(3))
    R         = exp([phi]x)      p       = J(phi) upsilon      (Eq. 4; J the SO(3) left Jacobian)
    w         = omega / |omega|                              (unit axis)
    a_perp    = a - <a, w> w                                 (Eq. 9, gauge-fixed anchor)
    p_par     = J(phi) <upsilon, w> w
    mean(t)   = R (mu - a_perp) + a_perp + p_par             (Eq. 10)
    Sigma(t)  = R Sigma R^T, i.e. quat(t) = quat(R) * quat

Where |omega| is too small to give an axis (ours: |omega|^2 <= 1e-12) the anchor
term is dropped (a_perp = a) and the whole translation is kept (p_par = p):
R is the identity there, so the splat just translates.  The two forms agree
exactly for a translation along the axis and differ for a perpendicular one,
which only the anchor can express; the paper's gauge fixing has the same seam.

The temporal term of the paper's Eq. 8 keeps growing linearly outside the
window; only the weight suppresses it.  Nothing here clamps dt.

The opacity weight w is the format's, not the paper's plain gamma (Eq. 1): Open4DGS's
runtime file calls it "smooth_cull_renormalized", and the animated-sog spec
(github.com/arrival-space/animated-sog, docs/format.md, MIT) gives it as
max((gamma - 0.05) / 0.95, 0): a splat is gone where gamma falls under 5% and the
rest is stretched back to 1 at the centre.  Training, scoring and every player use
it, so what is trained is what is shown.

Units: seconds for ``times`` and dt; ``zeta_*_r`` in radians per second and
``zeta_*_t`` in scene units per second.  Quaternions are wxyz here (the
.trigsbin file stores xyzw; trigsbin.py converts).  Scales are natural-log,
opacity is logit.  Needs torch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import Optional

import torch

#: |omega|^2 at or below which a splat is treated as not rotating (ours).
OMEGA_EPS = 1e-12
#: |phi| below which the series forms of exp and J are used.
SMALL_ANGLE = 1e-3
T_SIGMA_FLOOR = 1e-6

MOTION_KEYS = ('anchors', 'zeta_base_r', 'zeta_base_t', 'zeta_c0_r', 'zeta_c0_t',
               'zeta_c1_r', 'zeta_c1_t', 'zeta_c2_r', 'zeta_c2_t')


def skew(v: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] cross-product matrices of v [N, 3]."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
    ], dim=-2)


def _angle_terms(phi: torch.Tensor):
    """theta^2, and the series-safe coefficients A = sin(t)/t, B = (1-cos t)/t^2, C = (t - sin t)/t^3."""
    th2 = (phi * phi).sum(-1)
    small = th2 < SMALL_ANGLE ** 2
    safe = torch.where(small, torch.ones_like(th2), th2)
    th = torch.sqrt(safe)
    a = torch.where(small, 1.0 - th2 / 6.0 + th2 * th2 / 120.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0 + th2 * th2 / 720.0, (1.0 - torch.cos(th)) / safe)
    c = torch.where(small, 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0, (th - torch.sin(th)) / (safe * th))
    return th2, a, b, c


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation matrices [N, 3, 3] of rotation vectors phi [N, 3]."""
    _, a, b, _ = _angle_terms(phi)
    k = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(phi.shape[0], 3, 3)
    return eye + a[:, None, None] * k + b[:, None, None] * (k @ k)


def left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """The SO(3) left Jacobian J(phi) = I + B [phi]x + C [phi]x^2, [N, 3, 3]."""
    _, _, b, c = _angle_terms(phi)
    k = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(phi.shape[0], 3, 3)
    return eye + b[:, None, None] * k + c[:, None, None] * (k @ k)


def rotvec_to_quat_wxyz(phi: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (w first) of rotation vectors phi [N, 3]."""
    th2 = (phi * phi).sum(-1)
    small = th2 < SMALL_ANGLE ** 2
    safe = torch.where(small, torch.ones_like(th2), th2)
    th = torch.sqrt(safe)
    half = 0.5 * th
    w = torch.where(small, 1.0 - th2 / 8.0 + th2 * th2 / 384.0, torch.cos(half))
    k = torch.where(small, 0.5 - th2 / 48.0 + th2 * th2 / 3840.0, torch.sin(half) / th)
    return torch.cat([w[:, None], k[:, None] * phi], dim=-1)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b of wxyz quaternions [N, 4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


@dataclass
class TrigsTensors:
    """Per-splat parameters, float32 on one device.  N splats.

    means         [N, 3]    canonical position mu
    quats_wxyz    [N, 4]    canonical rotation (not necessarily normalised)
    log_scales    [N, 3]    natural-log scale
    opacity_logit [N]       peak opacity, logit
    sh            [N, K, 3] SH coefficients, DC at k = 0
    times         [N]       time centre, seconds
    log_sigma     [N]       log of the temporal sigma, seconds ("log_durations")
    anchors       [N, 3]    local anchor a
    zeta_base_r/t [N, 3]    base twist, omega and nu
    zeta_c0/c1/c2_r/t [N, 3]  the three Bezier control twists
    """
    means: torch.Tensor
    quats_wxyz: torch.Tensor
    log_scales: torch.Tensor
    opacity_logit: torch.Tensor
    sh: torch.Tensor
    times: torch.Tensor
    log_sigma: torch.Tensor
    anchors: torch.Tensor
    zeta_base_r: torch.Tensor
    zeta_base_t: torch.Tensor
    zeta_c0_r: torch.Tensor
    zeta_c0_t: torch.Tensor
    zeta_c1_r: torch.Tensor
    zeta_c1_t: torch.Tensor
    zeta_c2_r: torch.Tensor
    zeta_c2_t: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def sh_degree(self) -> int:
        k = int(self.sh.shape[1])
        degree = int(round(k ** 0.5)) - 1
        if (degree + 1) ** 2 != k:
            raise ValueError(f'sh has {k} coefficients per channel, not a square')
        return degree

    def slice(self, t):
        """(means [N, 3], quats_wxyz [N, 4], opacities [N]) at time t.  See slice_at()."""
        return slice_at(self, t)

    def to(self, device) -> 'TrigsTensors':
        return TrigsTensors(**{f.name: getattr(self, f.name).to(device) for f in dc_fields(self)})


def bezier_twist(base, c0, c1, c2, tau):
    """zeta(t) of Eqs. 7-8 for one half (rotation or translation) of the twist; tau is [N]."""
    b = 1.0 - tau
    return (base + (b * b)[:, None] * c0 + (2.0 * b * tau)[:, None] * c1 + (tau * tau)[:, None] * c2)


def slice_at(p: TrigsTensors, t):
    """Evaluate the model at time t: (means [N, 3], quats_wxyz [N, 4], opacities [N]).

    t is a Python float, a 0-d tensor, or an [N] tensor (one time per splat, which is how rolling
    shutter asks).  Differentiable in every tensor input.  The returned quaternions are unit-length
    rotations composed with the canonical ones; the canonical quaternion is left as given when a
    splat does not rotate.
    """
    sigma = torch.exp(p.log_sigma).clamp_min(T_SIGMA_FLOOR)
    dt = t - p.times
    tau = ((dt + 2.0 * sigma) / (4.0 * sigma)).clamp(0.0, 1.0)

    omega = bezier_twist(p.zeta_base_r, p.zeta_c0_r, p.zeta_c1_r, p.zeta_c2_r, tau)
    nu = bezier_twist(p.zeta_base_t, p.zeta_c0_t, p.zeta_c1_t, p.zeta_c2_t, tau)
    phi = omega * dt[:, None]
    ups = nu * dt[:, None]

    rot = so3_exp(phi)
    jac = left_jacobian(phi)

    w2 = (omega * omega).sum(-1)
    moving = w2 > OMEGA_EPS
    axis = omega / torch.sqrt(torch.where(moving, w2, torch.ones_like(w2)))[:, None]
    axis = torch.where(moving[:, None], axis, torch.zeros_like(axis))   # no axis: drops both axis terms below

    a_perp = p.anchors - (p.anchors * axis).sum(-1, keepdim=True) * axis
    ups_par = (ups * axis).sum(-1, keepdim=True) * axis
    ups_used = torch.where(moving[:, None], ups_par, ups)
    shift = (jac @ ups_used[..., None])[..., 0]

    centred = p.means - a_perp
    means = (rot @ centred[..., None])[..., 0] + a_perp + shift

    quats = quat_mul_wxyz(rotvec_to_quat_wxyz(phi), p.quats_wxyz)
    opacities = torch.sigmoid(p.opacity_logit) * temporal_weight(dt, sigma)
    return means, quats, opacities


#: Where the temporal gaussian is cut off and renormalised (the format's weight, see the module docstring).
WEIGHT_FLOOR = 0.05


def temporal_weight(dt, sigma):
    """The opacity weight of a splat at `dt` seconds from its centre, with temporal sigma `sigma`:
    max((gamma - 0.05) / 0.95, 0), gamma = exp(-0.5 (dt / sigma)^2).  1 at the centre, 0 from about 2.45 sigma out."""
    gamma = torch.exp(-0.5 * (dt / sigma) ** 2)
    return ((gamma - WEIGHT_FLOOR) / (1.0 - WEIGHT_FLOOR)).clamp_min(0.0)


def window_bounds(p: TrigsTensors):
    """(start, end) [N] of each splat's effective window [t_c - 2 s, t_c + 2 s] (Eq. 5)."""
    sigma = torch.exp(p.log_sigma).clamp_min(T_SIGMA_FLOOR)
    return p.times - 2.0 * sigma, p.times + 2.0 * sigma

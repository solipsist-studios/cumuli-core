# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""cumuli_core.trigs: the TRiGS slice (identity, pure motions, gauge, Jacobian, small angles, gradients)."""

import math

import pytest

torch = pytest.importorskip('torch')

from cumuli_core import trigs  # noqa: E402

D = torch.float64


def make(n=4, seed=0, **over):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=g, dtype=D) * sc)  # noqa: E731
    q = r(n, 4)
    fields = dict(
        means=r(n, 3), quats_wxyz=q / q.norm(dim=1, keepdim=True), log_scales=r(n, 3, sc=0.1) - 3,
        opacity_logit=r(n), sh=r(n, 4, 3, sc=0.1), times=torch.full((n,), 1.0, dtype=D),
        log_sigma=torch.full((n,), math.log(0.5), dtype=D), anchors=r(n, 3),
        zeta_base_r=r(n, 3, sc=0.3), zeta_base_t=r(n, 3, sc=0.3),
        zeta_c0_r=r(n, 3, sc=0.2), zeta_c0_t=r(n, 3, sc=0.2), zeta_c1_r=r(n, 3, sc=0.2), zeta_c1_t=r(n, 3, sc=0.2),
        zeta_c2_r=r(n, 3, sc=0.2), zeta_c2_t=r(n, 3, sc=0.2))
    fields.update(over)
    return trigs.TrigsTensors(**fields)


def zeroed(p, **keep):
    out = {}
    for k in trigs.MOTION_KEYS:
        out[k] = keep.get(k, torch.zeros_like(getattr(p, k)))
    return trigs.TrigsTensors(**{**{f: getattr(p, f) for f in ('means', 'quats_wxyz', 'log_scales', 'opacity_logit', 'sh', 'times', 'log_sigma')}, **out})


def test_the_splat_is_untouched_at_its_time_centre():
    p = make()
    means, quats, opac = trigs.slice_at(p, 1.0)
    assert torch.allclose(means, p.means, atol=1e-12)
    assert torch.allclose(quats, p.quats_wxyz, atol=1e-12)
    assert torch.allclose(opac, torch.sigmoid(p.opacity_logit), atol=1e-12)


def test_opacity_follows_the_temporal_gaussian():
    p = make()
    _, _, opac = trigs.slice_at(p, 1.5)    # one sigma away
    assert torch.allclose(opac, torch.sigmoid(p.opacity_logit) * math.exp(-0.5), atol=1e-12)


def test_a_pure_translation_moves_linearly():
    p = make()
    v = torch.tensor([0.3, -0.2, 0.5], dtype=D).expand(4, 3)
    q = zeroed(p, zeta_base_t=v.clone())
    means, quats, _ = trigs.slice_at(q, 1.4)
    assert torch.allclose(means, p.means + 0.4 * v, atol=1e-12)
    assert torch.allclose(quats, p.quats_wxyz, atol=1e-12)


def test_a_pure_rotation_turns_about_the_anchor_axis():
    p = make()
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=D)
    rate = 0.8
    q = zeroed(p, zeta_base_r=(rate * axis).expand(4, 3).clone())
    dt = 0.4
    means, quats, _ = trigs.slice_at(q, 1.0 + dt)
    th = rate * dt
    rz = torch.tensor([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=D)
    a_perp = q.anchors - (q.anchors @ axis)[:, None] * axis          # the anchor's part off the axis
    want = (rz @ (q.means - a_perp).T).T + a_perp
    assert torch.allclose(means, want, atol=1e-10)
    # the orientation turned by the same rotation
    r_got = quat_to_matrix(quats[0])
    r_can = quat_to_matrix(p.quats_wxyz[0])
    assert torch.allclose(r_got, rz @ r_can, atol=1e-10)


def test_moving_the_anchor_along_the_axis_changes_nothing():
    p = make()
    axis = torch.tensor([0.0, 0.6, 0.8], dtype=D)
    q = zeroed(p, zeta_base_r=(0.9 * axis).expand(4, 3).clone(), zeta_base_t=torch.randn(4, 3, dtype=D) * 0.1)
    moved = trigs.TrigsTensors(**{**q.__dict__, 'anchors': q.anchors + 2.3 * axis})
    a = trigs.slice_at(q, 1.35)[0]
    b = trigs.slice_at(moved, 1.35)[0]
    assert torch.allclose(a, b, atol=1e-10)


def test_the_bezier_hits_its_end_controls():
    n = 3
    base, c0, c1, c2 = (torch.randn(n, 3, dtype=D) for _ in range(4))
    assert torch.allclose(trigs.bezier_twist(base, c0, c1, c2, torch.zeros(n, dtype=D)), base + c0)
    assert torch.allclose(trigs.bezier_twist(base, c0, c1, c2, torch.ones(n, dtype=D)), base + c2)
    mid = trigs.bezier_twist(base, c0, c1, c2, torch.full((n,), 0.5, dtype=D))
    assert torch.allclose(mid, base + 0.25 * c0 + 0.5 * c1 + 0.25 * c2)


def test_the_bezier_clamps_outside_the_window_but_the_base_term_keeps_growing():
    p = make()
    q = zeroed(p, zeta_base_t=torch.ones(4, 3, dtype=D) * 0.2, zeta_c0_t=torch.ones(4, 3, dtype=D))
    far_a = trigs.slice_at(q, 1.0 + 10.0)[0]
    far_b = trigs.slice_at(q, 1.0 + 20.0)[0]
    # at 10 s and 20 s tau is 1 for both, so c0 contributes nothing; only the base grows with dt
    assert torch.allclose(far_b - far_a, 0.2 * 10.0 * torch.ones(4, 3, dtype=D), atol=1e-9)


def test_so3_exp_matches_the_matrix_exponential_across_the_small_angle_seam():
    for th in (0.0, 1e-9, 5e-4, 9.99e-4, 1.01e-3, 2e-3, 0.1, 1.0, 3.0):
        phi = torch.tensor([[0.3, -0.5, 0.8]], dtype=D)
        phi = phi / phi.norm() * th
        want = torch.linalg.matrix_exp(trigs.skew(phi))
        assert torch.allclose(trigs.so3_exp(phi), want, atol=1e-12), th


def test_the_left_jacobian_satisfies_its_defining_property():
    # exp(phi + d) = exp(J(phi) d) exp(phi) to first order in d
    for th in (1e-4, 0.5, 2.0):
        phi = torch.tensor([[0.4, 0.3, -0.7]], dtype=D)
        phi = phi / phi.norm() * th
        d = torch.tensor([[1.0, -2.0, 0.5]], dtype=D) * 1e-6
        lhs = trigs.so3_exp(phi + d)
        rhs = trigs.so3_exp((trigs.left_jacobian(phi) @ d[..., None])[..., 0]) @ trigs.so3_exp(phi)
        assert torch.allclose(lhs, rhs, atol=1e-10), th


def test_the_quaternion_of_a_rotation_vector_is_that_rotation():
    phi = torch.randn(5, 3, dtype=D)
    for scale in (1e-5, 0.4, 2.5):
        ph = phi * scale
        q = trigs.rotvec_to_quat_wxyz(ph)
        assert torch.allclose(q.norm(dim=1), torch.ones(5, dtype=D), atol=1e-12)
        for i in range(5):
            assert torch.allclose(quat_to_matrix(q[i]), trigs.so3_exp(ph[i:i + 1])[0], atol=1e-10)


def test_a_splat_that_does_not_rotate_translates_and_stays_finite_with_finite_gradients():
    p = make()
    v = torch.tensor([0.0, 0.5, 0.0], dtype=D).expand(4, 3).clone().requires_grad_(True)
    q = zeroed(p, zeta_base_t=v)
    q.anchors = q.anchors.clone().requires_grad_(True)
    means, quats, opac = trigs.slice_at(q, 1.3)
    assert torch.allclose(means, p.means + 0.3 * v.detach(), atol=1e-12)
    (means.sum() + quats.sum() + opac.sum()).backward()
    assert torch.isfinite(v.grad).all() and torch.isfinite(q.anchors.grad).all()


def test_a_translation_along_the_axis_is_continuous_as_the_rotation_vanishes():
    p = make()
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=D)
    out = []
    for w in (1e-3, 1e-5, 1e-7, 1e-9, 1e-13, 0.0):
        q = zeroed(p, zeta_base_r=(w * axis).expand(4, 3).clone(), zeta_base_t=(0.7 * axis).expand(4, 3).clone())
        out.append(trigs.slice_at(q, 1.5)[0])
    for m in out:
        assert torch.allclose(m, out[-1], atol=2e-3)
    assert torch.allclose(out[3], out[-1], atol=1e-8)


def test_each_splat_can_be_asked_for_its_own_time():
    p = make()
    t = torch.linspace(0.6, 1.4, 4, dtype=D)
    means, quats, opac = trigs.slice_at(p, t)
    for i in range(4):
        one = trigs.TrigsTensors(**{f: getattr(p, f)[i:i + 1] for f in p.__dataclass_fields__})
        m, q, o = trigs.slice_at(one, float(t[i]))
        assert torch.allclose(means[i], m[0], atol=1e-12) and torch.allclose(quats[i], q[0], atol=1e-12)
        assert torch.allclose(opac[i], o[0], atol=1e-12)


def test_gradients_reach_every_motion_parameter_and_the_anchor():
    p = make()
    for k in trigs.MOTION_KEYS:
        setattr(p, k, getattr(p, k).clone().requires_grad_(True))
    means, quats, opac = trigs.slice_at(p, 1.2)
    (means.pow(2).sum() + quats.sum() + opac.sum()).backward()
    for k in trigs.MOTION_KEYS:
        g = getattr(p, k).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, k


def quat_to_matrix(q):
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=D)

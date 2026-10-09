# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.spacetime: the temporal slice and the PLY <-> tensor mapping."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import _fixture as fixture  # noqa: E402
from cumuli_core.ply import read_sogst_ply, write_sogst_ply  # noqa: E402
from cumuli_core.spacetime import from_fields, slice_at, to_fields  # noqa: E402


def lower_bands(f_rest, coeffs):
    n = f_rest.shape[0]
    return np.ascontiguousarray(
        f_rest.reshape(n, 3, 15)[:, :, :coeffs].reshape(n, 3 * coeffs))


def random_params(n=7, dtype=torch.float64, accel=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    p = dict(xyz=r(n, 3), velocity=r(n, 3), opacity_logit=r(n),
             t_center=r(n) * 0.3, t_sigma=r(n).abs() * 0.2 + 0.05)
    p["accel"] = r(n, 3) if accel else None
    return p


# --------------------------------------------------------------------------
# slice_at
# --------------------------------------------------------------------------

@pytest.mark.parametrize("with_accel", [False, True])
def test_slice_matches_the_formula(with_accel):
    p = random_params(accel=with_accel)
    t = 0.37
    means, alpha = slice_at(p["xyz"], p["velocity"], p["opacity_logit"],
                            p["t_center"], p["t_sigma"], t, accel=p["accel"])
    x = {k: (v.numpy() if v is not None else None) for k, v in p.items()}
    dt = t - x["t_center"]
    ref = x["xyz"] + x["velocity"] * dt[:, None]
    if with_accel:
        ref = ref + x["accel"] * (dt * dt)[:, None]
    ref_a = 1 / (1 + np.exp(-x["opacity_logit"])) * np.exp(-0.5 * (dt / x["t_sigma"]) ** 2)
    np.testing.assert_allclose(means.numpy(), ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alpha.numpy(), ref_a, rtol=1e-12, atol=1e-12)


def test_alpha_peaks_at_t_center_and_mean_is_xyz_there():
    p = random_params(n=1)
    tc = p["t_center"][0]
    means, alpha = slice_at(p["xyz"], p["velocity"], p["opacity_logit"],
                            p["t_center"], p["t_sigma"], tc, accel=p["accel"])
    torch.testing.assert_close(means, p["xyz"])
    torch.testing.assert_close(alpha, torch.sigmoid(p["opacity_logit"]))


def test_t_may_be_a_zero_dim_tensor():
    p = random_params()
    a = slice_at(p["xyz"], p["velocity"], p["opacity_logit"], p["t_center"],
                 p["t_sigma"], 0.25, accel=p["accel"])
    b = slice_at(p["xyz"], p["velocity"], p["opacity_logit"], p["t_center"],
                 p["t_sigma"], torch.tensor(0.25, dtype=torch.float64), accel=p["accel"])
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


@pytest.mark.parametrize("with_accel", [False, True])
def test_slice_gradients_pass_gradcheck(with_accel):
    """Every parameter, and t itself, gets correct analytic gradients."""
    p = random_params(n=5, accel=with_accel)
    names = ["xyz", "velocity", "opacity_logit", "t_center", "t_sigma"]
    t = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    inputs = [p[k].clone().requires_grad_(True) for k in names]
    if with_accel:
        inputs.append(p["accel"].clone().requires_grad_(True))

    def fn(xyz, vel, op, tc, ts, tt, *acc):
        return slice_at(xyz, vel, op, tc, ts, tt, accel=acc[0] if acc else None)

    assert torch.autograd.gradcheck(fn, (*inputs[:5], t, *inputs[5:]))


def test_slice_does_not_modify_its_inputs():
    p = random_params()
    before = {k: v.clone() for k, v in p.items()}
    leaves = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    means, alpha = slice_at(leaves["xyz"], leaves["velocity"], leaves["opacity_logit"],
                            leaves["t_center"], leaves["t_sigma"], 0.5,
                            accel=leaves["accel"])
    (means.sum() + alpha.sum()).backward()
    for k in p:
        torch.testing.assert_close(leaves[k].detach(), before[k])
        assert leaves[k].grad is not None and torch.isfinite(leaves[k].grad).all()


# --------------------------------------------------------------------------
# from_fields / to_fields
# --------------------------------------------------------------------------

def eval_render_shs(fields, n_sh):
    """The SH tensor exactly as cumuli's eval_render.py builds it."""
    n = len(fields["x"])
    shs = torch.zeros((n, n_sh, 3), dtype=torch.float32)
    for c in range(3):
        shs[:, 0, c] = torch.tensor(np.asarray(fields[f"f_dc_{c}"]), dtype=torch.float32)
    if n_sh > 1:
        f_rest = torch.tensor(np.ascontiguousarray(fields["f_rest"]), dtype=torch.float32)
        shs[:, 1:, :] = f_rest.reshape(n, 3, n_sh - 1).permute(0, 2, 1)
    return shs


@pytest.mark.parametrize("coeffs, degree", [(0, 0), (3, 1), (8, 2), (15, 3)])
def test_sh_layout_matches_eval_render(coeffs, degree):
    fields, _ = fixture.build_fixture(count=40, include_sh=True, seed=2)
    if coeffs:
        fields["f_rest"] = lower_bands(fields["f_rest"], coeffs)
    else:
        del fields["f_rest"]
    st = from_fields(fields)
    assert st.sh_degree == degree
    torch.testing.assert_close(st.sh, eval_render_shs(fields, (degree + 1) ** 2))


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("include_sh", [False, True])
def test_fields_round_trip_through_tensors_and_ply(tmp_path, degree, include_sh):
    fields, meta = fixture.build_fixture(count=50, degree=degree,
                                         include_sh=include_sh, seed=4)
    path = tmp_path / "s.ply"
    write_sogst_ply(path, fields, meta["time_min"], meta["time_max"], meta["fps"])
    _, read = read_sogst_ply(path)

    st = from_fields(read)
    assert (st.accel is not None) is (degree == 2)
    back = to_fields(st)
    assert set(back) == set(read)
    for k in read:
        np.testing.assert_array_equal(back[k], read[k], err_msg=k)

    write_sogst_ply(tmp_path / "again.ply", back, 0.0, 2.0, 30.0)


def test_from_fields_values():
    fields, _ = fixture.build_fixture(count=12, degree=2, include_sh=False, seed=9)
    st = from_fields(fields)
    np.testing.assert_allclose(st.xyz[:, 1].numpy(), fields["y"], rtol=1e-6)
    np.testing.assert_allclose(st.quats_wxyz[:, 0].numpy(), fields["rot_0"], rtol=1e-6)
    np.testing.assert_allclose(st.scales[:, 2].numpy(), np.exp(fields["scale_2"]), rtol=1e-5)
    np.testing.assert_allclose(st.accel[:, 1].numpy(), fields["ay"], rtol=1e-6)
    assert st.count == 12 and st.sh.shape == (12, 1, 3)
    assert all(t.dtype == torch.float32 for t in (st.xyz, st.t_sigma, st.sh))
    means, alpha = st.slice(0.5)
    assert means.shape == (12, 3) and alpha.shape == (12,)

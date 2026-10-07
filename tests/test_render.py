# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.render on the GPU (skipped without CUDA and gsplat)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from cumuli_core.cameras import Camera  # noqa: E402
from cumuli_core.render import camera_tensors, render  # noqa: E402
from cumuli_core.sh import rgb_to_sh0  # noqa: E402
from cumuli_core.spacetime import slice_at  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda")
W, H = 64, 48


def camera(z=3.0):
    w2c = np.eye(4)
    w2c[2, 3] = z                            # world origin at depth z
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    return Camera(name="c", image_path="c", w2c=w2c, K=K, width=W, height=H, time=0.0)


def one_splat(colour=(1.0, 0.0, 0.0)):
    means = torch.zeros(1, 3, device=DEV)
    quats = torch.tensor([[1.0, 0, 0, 0]], device=DEV)
    scales = torch.full((1, 3), 0.2, device=DEV)
    sh = torch.tensor(rgb_to_sh0(np.array(colour)), dtype=torch.float32,
                      device=DEV).reshape(1, 1, 3)
    return means, quats, scales, sh


def test_centred_splat_renders_at_the_principal_point():
    means, quats, scales, sh = one_splat()
    rgb, alpha, info = render(means, quats, scales, torch.ones(1, device=DEV), sh,
                              [camera()], W, H, sh_degree=0)
    assert rgb.shape == (1, H, W, 3) and alpha.shape == (1, H, W, 1)
    a = alpha[0, ..., 0].cpu().numpy()
    y, x = np.unravel_index(np.argmax(a), a.shape)
    assert abs(x - (W / 2 - 0.5)) <= 1 and abs(y - (H / 2 - 0.5)) <= 1
    centre = rgb[0, H // 2, W // 2].cpu().numpy()
    assert centre[0] > 0.9 and centre[1] < 0.05


def test_tensor_cameras_equal_camera_objects():
    means, quats, scales, sh = one_splat()
    op = torch.ones(1, device=DEV)
    a, _, _ = render(means, quats, scales, op, sh, [camera()], W, H, sh_degree=0)
    vm, ks = camera_tensors([camera()], DEV)
    b, _, _ = render(means, quats, scales, op, sh, (vm, ks), W, H, sh_degree=0)
    torch.testing.assert_close(a, b)


def test_background_fills_where_alpha_is_zero():
    means, quats, scales, sh = one_splat()
    bg = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    rgb, alpha, _ = render(means, quats, scales, torch.ones(1, device=DEV), sh,
                           [camera()], W, H, sh_degree=0, background=bg)
    torch.testing.assert_close(rgb[0, 0, 0], bg)


def test_gradients_flow_from_pixels_to_spacetime_parameters():
    means0, quats, scales, sh = one_splat()
    xyz = means0.clone().requires_grad_(True)
    vel = torch.tensor([[0.1, 0.0, 0.0]], device=DEV, requires_grad=True)
    op = torch.tensor([2.0], device=DEV, requires_grad=True)
    tc = torch.tensor([0.0], device=DEV, requires_grad=True)
    ts = torch.tensor([0.5], device=DEV, requires_grad=True)
    means, opac = slice_at(xyz, vel, op, tc, ts, 0.2)
    rgb, _, _ = render(means, quats, scales, opac, sh, [camera()], W, H, sh_degree=0)
    rgb.sum().backward()
    for p in (xyz, vel, op, tc, ts):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0

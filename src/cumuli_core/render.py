# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.render - a thin wrapper over gsplat.rasterization.

render() takes splats already sliced to one instant (see
cumuli_core.spacetime.slice_at) plus a batch of cameras, and returns RGB,
alpha and gsplat's info dict.  It adds two things over calling gsplat
directly: Camera objects are converted to view and intrinsic matrices, and
an optional background colour is composited under the alpha.

Needs torch and gsplat.
"""

import numpy as np
import torch


def camera_tensors(cameras, device):
    """(viewmats [B, 4, 4], Ks [B, 3, 3]) float32 on `device` from a list of
    cumuli_core.cameras.Camera."""
    vm = np.stack([np.asarray(c.w2c, dtype=np.float64) for c in cameras])
    ks = np.stack([np.asarray(c.K, dtype=np.float64) for c in cameras])
    return (torch.tensor(vm, dtype=torch.float32, device=device),
            torch.tensor(ks, dtype=torch.float32, device=device))


def render(means, quats_wxyz, scales, opacities, sh, cameras, width, height,
           sh_degree, background=None, packed=False, **gsplat_kwargs):
    """Rasterize one batch of views.

    means [N, 3], quats_wxyz [N, 4], scales [N, 3] (linear, not log),
    opacities [N] (in [0, 1], already time-windowed), sh [N, K, 3].

    cameras is a list of Camera (all rendered at width x height) or a
    (viewmats [B, 4, 4], Ks [B, 3, 3]) tuple of tensors.  sh_degree may be
    lower than the degree sh carries.

    background, if given, is a [3] tensor composited as
    rgb + (1 - alpha) * background.  Without it the colour is gsplat's
    premultiplied output, which is the same as compositing over black.

    Extra keyword arguments go straight to gsplat.rasterization (near_plane,
    rasterize_mode, absgrad, ...).  Returns (rgb [B, H, W, 3],
    alpha [B, H, W, 1], info).  rgb is not clamped.
    """
    from gsplat import rasterization

    if isinstance(cameras, (tuple,)) and len(cameras) == 2 and torch.is_tensor(cameras[0]):
        viewmats, Ks = cameras
    else:
        viewmats, Ks = camera_tensors(cameras, means.device)
    gsplat_kwargs.setdefault('render_mode', 'RGB')
    rgb, alpha, info = rasterization(
        means, quats_wxyz, scales, opacities, sh, viewmats, Ks,
        int(width), int(height), sh_degree=sh_degree, packed=packed,
        **gsplat_kwargs)
    if background is not None:
        rgb = rgb + (1.0 - alpha) * background.to(rgb).reshape(1, 1, 1, -1)
    return rgb, alpha, info

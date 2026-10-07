# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.metrics - PSNR, SSIM and LPIPS on [0, 1] images.

Images are HxWx3 or BxHxWx3 float tensors in [0, 1], channels last, the
layout render() returns.  Each function returns a 0-d tensor: the mean over
the batch for SSIM and LPIPS, and PSNR of the mean squared error over every
pixel of the batch.  For a single image that is exactly what eval_render.py
has always reported.

SSIM is torchmetrics' structural_similarity_index_measure with
data_range=1.0.  LPIPS is the `lpips` package's AlexNet variant, fed
inputs scaled to [-1, 1].  Report LPIPS beside PSNR: on a masked subject
PSNR is dominated by background that every model renders perfectly.

Needs torch, torchmetrics and lpips.
"""

import torch


def _bhwc(img):
    if img.dim() == 3:
        return img.unsqueeze(0)
    if img.dim() != 4:
        raise ValueError(f'expected HxWxC or BxHxWxC, got shape {tuple(img.shape)}')
    return img


def _bchw(img):
    # A single image goes through permute(2, 0, 1)[None], exactly as
    # eval_render.py always did.  The batched route yields a tensor with a
    # different batch stride, and cuDNN then picks a different convolution
    # algorithm: LPIPS moved by about 1e-6 per view.
    if img.dim() == 3:
        return img.permute(2, 0, 1)[None]
    return _bhwc(img).permute(0, 3, 1, 2)


def psnr(img, gt):
    """Peak signal-to-noise ratio in dB, data range 1."""
    mse = torch.mean((img - gt) ** 2)
    return -10.0 * torch.log10(mse)


def ssim(img, gt):
    """Structural similarity (torchmetrics, gaussian window, data range 1)."""
    from torchmetrics.functional import structural_similarity_index_measure
    return structural_similarity_index_measure(_bchw(img), _bchw(gt), data_range=1.0)


class LPIPS:
    """Learned perceptual distance, AlexNet backbone.

    The network loads on first call, on the device of the first input (or
    `device`), so constructing this costs nothing.  Its weights are frozen.
    verbose=True keeps the lpips package's setup messages.

        lpips = LPIPS()
        d = lpips(render, gt)      # 0-d tensor, mean over the batch
    """

    def __init__(self, net='alex', device=None, verbose=False):
        self.net_name = net
        self.device = device
        self.verbose = verbose
        self._net = None

    def _load(self, device):
        import lpips as lpips_mod
        net = lpips_mod.LPIPS(net=self.net_name, verbose=self.verbose).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._net = net

    def __call__(self, img, gt):
        if self._net is None:
            self._load(self.device or img.device)
        x = _bchw(img) * 2 - 1
        y = _bchw(gt) * 2 - 1
        return self._net(x, y).mean()

# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core.metrics against the exact calls eval_render.py makes."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

from cumuli_core.metrics import LPIPS, psnr, ssim  # noqa: E402


def images(seed=0, b=None, h=48, w=64):
    g = torch.Generator().manual_seed(seed)
    shape = (h, w, 3) if b is None else (b, h, w, 3)
    gt = torch.rand(*shape, generator=g)
    img = (gt + 0.05 * torch.randn(*shape, generator=g)).clamp(0, 1)
    return img, gt


def test_psnr_matches_the_definition():
    img, gt = images()
    mse = torch.mean((img - gt) ** 2)
    assert float(psnr(img, gt)) == pytest.approx(float(-10 * torch.log10(mse)), abs=1e-5)
    assert float(psnr(torch.full((4, 4, 3), 0.5), torch.full((4, 4, 3), 0.6))) == \
        pytest.approx(20.0, abs=1e-4)


def test_ssim_matches_torchmetrics_on_chw():
    from torchmetrics.functional import structural_similarity_index_measure as tm
    img, gt = images()
    ref = tm(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0)
    assert float(ssim(img, gt)) == pytest.approx(float(ref), abs=1e-7)
    assert float(ssim(gt, gt)) == pytest.approx(1.0, abs=1e-6)


def test_batched_ssim_is_the_mean_of_singles():
    img, gt = images(b=2)
    singles = [float(ssim(img[i], gt[i])) for i in range(2)]
    assert float(ssim(img, gt)) == pytest.approx(sum(singles) / 2, abs=1e-6)


def test_bad_rank_is_refused():
    with pytest.raises(ValueError):
        ssim(torch.zeros(3, 3), torch.zeros(3, 3))


@torch.no_grad()
def test_lpips_matches_the_direct_call():
    lpips_mod = pytest.importorskip("lpips")
    img, gt = images()
    ref_net = lpips_mod.LPIPS(net="alex", verbose=False)
    chw, gt_chw = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    ref = float(ref_net(chw * 2 - 1, gt_chw * 2 - 1))
    metric = LPIPS()
    assert float(metric(img, gt)) == pytest.approx(ref, abs=1e-6)
    assert float(metric(gt, gt)) == pytest.approx(0.0, abs=1e-6)
    b_img, b_gt = images(b=2)
    singles = [float(metric(b_img[i], b_gt[i])) for i in range(2)]
    assert float(metric(b_img, b_gt)) == pytest.approx(sum(singles) / 2, abs=1e-6)

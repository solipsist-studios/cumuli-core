# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""cumuli_core: shared components of the Cumuli 4D Gaussian splatting stack.

Modules:

    ply        4D interchange PLY reader, writer and CLI (numpy)
    sh         spherical-harmonic block widths and band counts (numpy)
    sogst      .sogst archive decoder (numpy + pillow)
    cameras    transforms.json camera loader (numpy)
    dataset    dataset_4dgs/ reader (numpy + pillow)
    spacetime  per-splat tensors and the temporal slice (torch)
    render     gsplat rasterization wrapper (torch + gsplat)
    metrics    PSNR, SSIM, LPIPS (torch + torchmetrics + lpips)

Nothing is imported here, so `import cumuli_core.ply` never pulls in torch.
"""

__version__ = '0.1.0'

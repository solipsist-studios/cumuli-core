# cumuli-core

Shared Python components of the Cumuli 4D Gaussian splatting stack. The
cumuli pipeline scripts and the cumuli trainer both import them.

| Module | Needs | What it holds |
|---|---|---|
| `cumuli_core.ply` | numpy | 4D interchange PLY reader, writer and CLI |
| `cumuli_core.sh` | nothing | SH block widths and band counts |
| `cumuli_core.sogst` | numpy, pillow | .sogst archive decoder, `load_model()` |
| `cumuli_core.cameras` | numpy | `Camera`, `load_transforms()`, `group_by_time()` |
| `cumuli_core.shutter` | numpy, torch (optional) | rolling-shutter geometry: raw sensor row fractions, per-Gaussian readout times |
| `cumuli_core.dataset` | numpy, pillow | `Dataset4D` for a `dataset_4dgs/` directory |
| `cumuli_core.spacetime` | torch | `SpacetimeTensors`, `slice_at()`, PLY field mapping |
| `cumuli_core.render` | torch, gsplat | `render()`, a thin `gsplat.rasterization` wrapper |
| `cumuli_core.metrics` | torch, torchmetrics, lpips | `psnr()`, `ssim()`, `LPIPS` |

The PLY format is specified in `docs/sogst-format.md` section 7 of the cumuli
repo (CC-BY-4.0).

## Install

```bash
pip install -e .            # numpy modules only
pip install -e ".[torch]"   # rendering and metrics too
```

In the cumuli conda env, install with `--no-deps` so torch and gsplat stay
at the env's pins. cumuli's `scripts/setup_cumuli_env.sh` does this from the
`deps/cumuli-core` submodule.

## Example

```python
from cumuli_core.cameras import load_transforms
from cumuli_core.render import render
from cumuli_core.sogst import load_model
from cumuli_core.spacetime import from_fields

header, fields = load_model('splat_4d.sogst')        # or a 4D interchange .ply
st = from_fields(fields, 'cuda')
cam = load_transforms('dataset_4dgs/transforms_test.json')[0]
means, opacities = st.slice(header['time_min'] + cam.time)
rgb, alpha, _ = render(means, st.quats_wxyz, st.scales, opacities, st.sh,
                       [cam], cam.width, cam.height, st.sh_degree)
```

Rendering and scoring this way matches cumuli's `eval_render.py` exactly. On
`synth_runs/dur_12` (48 views), PSNR 39.3599, SSIM 0.99286 and LPIPS
0.006639 are bit-identical between the two.

## Tests

```bash
pytest tests
```

The GPU tests in `tests/test_render.py` skip without CUDA. Everything else
runs on CPU.

## Attributions

No third-party source code is included.

- **Formats:** the 4D interchange PLY and the `.sogst` container are specified in
  the cumuli repo (`docs/sogst-format.md`, CC-BY-4.0). The `.sogst` container
  extends PlayCanvas's SOG format (MIT, https://github.com/playcanvas/splat-transform).
  The per-splat PLY conventions (spherical-harmonic coefficients, logit opacity,
  log scales) are those of Kerbl, Kopanas, Leimkühler and Drettakis, "3D Gaussian
  Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023. The temporal
  model (polynomial motion, temporal radial-basis opacity) follows Li, Chen, Li and
  Xu, "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis",
  CVPR 2024.
- **Rolling shutter (`cumuli_core.shutter`):** the raw `OPENCV_FISHEYE` and `OPENCV`
  camera models are OpenCV's, as documented (Apache-2.0); the tests use OpenCV's
  `cv2.fisheye.distortPoints` and `cv2.projectPoints` as the numerical reference.
  The mid-line time convention and the mapping from an undistorted pixel to a raw
  sensor row were informed by the documentation and a few source excerpts of
  Open4DGS-Core (MIT, https://open4dgs.asunemlabs.com) and Open4DGS-Trigs (AGPL-3.0,
  https://github.com/ASU-NEM-Labs/open4dgs-trigs). No code from either is included.
- **Dependencies (not vendored):** numpy and PyTorch (BSD-3-Clause), gsplat and
  torchmetrics (Apache-2.0), lpips (BSD-2-Clause), pillow (MIT-CMU), OpenCV for the
  tests (Apache-2.0).

## Licence

PolyForm Noncommercial License 1.0.0 (see `LICENSE.md`).
Commercial licences: https://solipsist.studio
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

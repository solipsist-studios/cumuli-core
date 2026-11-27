# Changelog

## 0.2.0

- `cumuli_core.trigs`: the TRiGS motion model (Open4DGS `trigs.se3_bezier_v1`) and its temporal slice, written
  from the TRiGS paper (arXiv 2604.00538): temporal opacity window, quadratic Bezier se(3) twist, and the
  anchor-centred rotation and translation (`slice_at`, `so3_exp`, `left_jacobian`). Accepts one time or one
  time per splat (rolling shutter).
- `cumuli_core.trigsbin`: reads Open4DGS-Core's `.trigsbin` runtime file (MIT packager, runtime format 1):
  `read`, and `TrigsModel`/`load_model` to slice it at a global time with the clips' fades.

New per-view fields on `Camera`, each read from the matching key of a
`transforms_train.json` or `transforms_test.json` entry. All have defaults,
so existing datasets load unchanged.

- `loss_weight` (default 1, at least 0): how much the view's colour loss
  counts in training. Negative, non-finite or non-numeric values raise
  `ValueError`, naming the file and the view.
- `time_offset` (default 0) and `capture_time`: a camera's sub-frame capture
  offset in seconds. Views still group by the nominal `time`.
- `readout_s` (default 0) and `raw_camera`: the rolling-shutter readout and the
  raw sensor camera model. The new `cumuli_core.shutter` maps undistorted
  pixels to raw sensor row fractions and gives each Gaussian the time of the
  raw row its mean projects to.
- `visit_weight` (default 1, positive): how many times as often training
  should visit the view's instant.

## 0.1.0

First release: 4D interchange PLY, `.sogst` decoder, cameras, dataset reader,
spacetime slice, gsplat render wrapper and metrics.

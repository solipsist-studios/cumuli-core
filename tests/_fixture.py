# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""The synthetic conformance fixture's field builder, copied from cumuli's
scripts/make_sogst_fixture.py so the PLY tests run without cumuli.

A ballistic parabola: splats launched from near the origin under constant
downward acceleration, each fading in and out around its own t_center.
Quaternions hit all four smallest-three cases by construction.
"""

import numpy as np

FIXTURE_SEED = 20260813
GRAVITY = -9.81
SEGMENT_DURATION = 0.1
GAP_EDGE_NUDGE = 1e-3
BLOCK_SPACING = 1.0
BLOCK_RADIUS = 0.12


def build_fixture(count=8192, time_min=0.0, time_max=2.0, degree=2,
                  include_sh=True, persistent_fraction=0.08, seed=FIXTURE_SEED,
                  gap=None, blocks=False):
    """Build the parabola fixture's field arrays.

    `gap` is an optional (start, end) window in seconds that no splat's
    t_center falls into, which forces the encoder to emit empty segments
    (`first == last`).  Those are spec-legal (section 5) and no real capture
    produces them, so the branches that handle them (a player's culling
    scan, an encoder's zero-splat group) are otherwise only ever tested by
    reading the code.

    `blocks` replaces the parabola with static, spatially separated clusters,
    one per temporal segment plus one for the persistent group.  It exists
    because "nothing broke" is not the same claim as "the right segments were
    drawn", and the parabola cannot tell them apart: it is an expanding cloud,
    so a player that culls nothing at all looks the same as one that culls
    correctly.  With `blocks` the drawn population is countable by eye.  At
    any instant you should see the persistent block plus only the block(s)
    whose segment spans that instant.  A culling bug shows up as a block
    that is present when it should not be (or missing when it should).
    Combined with `gap` it is the strong form: an empty RUN whose surviving
    neighbours are in visibly different places.

    Motion is deliberately zero here (degree 1, velocity 0): a moving block
    would reintroduce the ambiguity the mode exists to remove.  Section 4.8
    forbids synthesising `accel`, so block mode never emits it.

    Returns (fields, meta) where `meta` records the analytic constants a
    verifier may want (the gravity term, the persistent split, the block
    layout) alongside the clip scalars.
    """
    rng = np.random.default_rng(seed)
    n = int(count)

    # Temporal layout: most splats are short-lived and spread evenly across
    # the clip, so segmentation has something to segment.  A minority get a
    # long sigma and must end up in the persistent range.
    # The short sigmas are capped so that a short-lived splat's active span
    # (2 * k_sigma * sigma, k_sigma = 3.8) stays under the default
    # persistence threshold of 3 * 0.1 s.  Otherwise the encoder promotes
    # most of them to persistent and the fixture stops exercising
    # per-segment culling, which is the thing it is there to exercise.
    t_center = rng.uniform(time_min, time_max, n)
    if gap is not None:
        # Fold anything inside the window out to its edges, keeping the
        # splat count exact.
        lo, hi = float(gap[0]), float(gap[1])
        mid = 0.5 * (lo + hi)
        inside = (t_center > lo) & (t_center < hi)
        # Nudged clear of the edge, not onto it. A gap bound is typically a
        # multiple of segment_duration, so folding *onto* it parks a large
        # plateau of splats exactly on a bucket boundary.  `t/duration`
        # then buckets one way in float64 and the other after the float32 PLY
        # round-trip, moving the whole plateau between two adjacent segments.
        # That made the fixture's own .ply and .sogst disagree about 1,218
        # splats. The nudge is ~4 orders above float32 epsilon here and 1% of
        # a default bucket, and it widens the empty window rather than
        # narrowing it, so the gap stays a gap.
        t_center[inside & (t_center <= mid)] = lo - GAP_EDGE_NUDGE
        t_center[inside & (t_center > mid)] = hi + GAP_EDGE_NUDGE
    t_sigma = rng.uniform(0.01, 0.03, n)
    n_persistent = int(round(persistent_fraction * n))
    persistent_idx = rng.choice(n, size=n_persistent, replace=False)
    t_sigma[persistent_idx] = rng.uniform(0.4, 0.7, n_persistent)

    # Ballistic motion.  Each splat is a particle launched at t = 0 from
    # near the origin.  Its PLY columns describe that trajectory re-anchored
    # to its own t_center, which is exactly the re-anchoring a producer has
    # to get right (position, velocity and accel move together).
    launch = rng.normal(0.0, 0.05, (n, 3))
    speed = rng.uniform(2.0, 6.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    elevation = rng.uniform(np.pi / 6, np.pi / 3, n)
    v0 = np.stack([
        speed * np.cos(elevation) * np.cos(theta),
        speed * np.sin(elevation),
        speed * np.cos(elevation) * np.sin(theta),
    ], axis=1)
    accel_true = np.zeros((n, 3))
    accel_true[:, 1] = 0.5 * GRAVITY        # raw dt^2 coefficient, NOT a/2 again

    # Re-anchor to t_center:  p(t) = p0 + v0*t + a*t^2  =>  at tc,
    #   xyz = p(tc),  v = v0 + 2*a*tc,  a unchanged.
    tc = t_center[:, None]
    xyz = launch + v0 * tc + accel_true * tc * tc
    velocity = v0 + 2.0 * accel_true * tc

    block_layout = None
    if blocks:
        # One cluster per segment the packer will bucket into, laid out along
        # x in segment order, plus the persistent group lifted clear on y.
        # The bucket expression mirrors compute_sogst_order exactly.  If the
        # two ever drift, the blocks stop lining up with the segments and the
        # fixture silently stops testing what it claims to.
        n_seg = max(1, int(np.ceil((time_max - time_min) / SEGMENT_DURATION)))
        bucket = np.clip(((t_center - time_min) / SEGMENT_DURATION).astype(np.int64),
                         0, n_seg - 1)
        centres = np.zeros((n, 3))
        centres[:, 0] = (bucket - 0.5 * (n_seg - 1)) * BLOCK_SPACING
        is_persistent = np.zeros(n, dtype=bool)
        is_persistent[persistent_idx] = True
        centres[is_persistent, 0] = 0.0
        centres[is_persistent, 1] = BLOCK_SPACING
        xyz = centres + rng.normal(0.0, BLOCK_RADIUS / 3.0, (n, 3))
        velocity = np.zeros((n, 3))
        accel_true = np.zeros((n, 3))
        block_layout = {
            'segment_duration': float(SEGMENT_DURATION),
            'segments': int(n_seg),
            'spacing': float(BLOCK_SPACING),
            'radius': float(BLOCK_RADIUS),
            'segment_block_x': [float((k - 0.5 * (n_seg - 1)) * BLOCK_SPACING)
                                for k in range(n_seg)],
            'persistent_block': [0.0, float(BLOCK_SPACING), 0.0],
            'splats_per_segment': [int((bucket[~is_persistent] == k).sum())
                                   for k in range(n_seg)],
        }

    # Quaternions hitting every smallest-three case by construction: cycle
    # the largest-magnitude component through w, x, y, z.
    quat = rng.normal(0.0, 1.0, (n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dominant = np.arange(n) % 4
    boost = np.zeros((n, 4))
    boost[np.arange(n), dominant] = 2.0
    quat = quat + boost
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    log_scales = rng.uniform(-6.0, -2.0, (n, 3))
    opacity_logit = rng.uniform(-1.0, 4.0, n)
    f_dc = rng.uniform(-1.5, 1.5, (n, 3))

    fields = {
        'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
        'rot_0': quat[:, 0], 'rot_1': quat[:, 1],
        'rot_2': quat[:, 2], 'rot_3': quat[:, 3],
        'scale_0': log_scales[:, 0], 'scale_1': log_scales[:, 1],
        'scale_2': log_scales[:, 2],
        'opacity': opacity_logit,
        'f_dc_0': f_dc[:, 0], 'f_dc_1': f_dc[:, 1], 'f_dc_2': f_dc[:, 2],
        'vx': velocity[:, 0], 'vy': velocity[:, 1], 'vz': velocity[:, 2],
        't_center': t_center,
        't_sigma': t_sigma,
    }
    if degree == 2 and not blocks:
        fields['ax'] = accel_true[:, 0]
        fields['ay'] = accel_true[:, 1]
        fields['az'] = accel_true[:, 2]
    if include_sh:
        # Channel-major, index j*15 + k, decaying by band so the values look
        # like a real SH tail.  Crucially it is also spatially coherent
        # (a smooth function of launch direction plus a little noise),
        # because real SH is, and because pure per-splat noise is the
        # worst possible case for the vector quantizer: it makes the
        # encoder's VQ error look like a layout bug in a field comparison.
        band_scale = np.tile(np.repeat([0.4, 0.15, 0.05], [3, 5, 7]), 3)[None, :]
        direction = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
        basis = np.stack([direction[:, 0], direction[:, 1], direction[:, 2],
                          direction[:, 0] * direction[:, 1]], axis=1)   # [n, 4]
        mixing = rng.normal(0.0, 1.0, (4, 45))
        fields['f_rest'] = (basis @ mixing + rng.normal(0.0, 0.15, (n, 45))) * band_scale

    meta = {
        'time_min': float(time_min), 'time_max': float(time_max),
        'fps': 30.0, 'count': n,
        'motion_degree': 1 if blocks else int(degree),
        'gap': (None if gap is None else [float(gap[0]), float(gap[1])]),
        'gravity_dt2_coefficient': (None if blocks else float(0.5 * GRAVITY)),
        'long_lived_splats': int(n_persistent),
        'seed': int(seed),
        'blocks': block_layout,
    }
    return fields, meta

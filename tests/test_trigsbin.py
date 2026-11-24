# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""cumuli_core.trigsbin: a checkpoint packed by Open4DGS-Core's packager reads back and slices alike.

The round trip needs Open4DGS-Core's `deliver/make-trigs-timeline.py` (MIT): set OPEN4DGS_CORE_ROOT, or have a
checkout at ~/Dev/github/open4dgs-core.  Without it those tests are skipped; the header tests need nothing.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from cumuli_core import trigs, trigsbin  # noqa: E402

PACKER = Path(os.environ.get('OPEN4DGS_CORE_ROOT') or Path.home() / 'Dev/github/open4dgs-core') / 'deliver' / 'make-trigs-timeline.py'
needs_packer = pytest.mark.skipif(not PACKER.is_file(), reason='Open4DGS-Core is not checked out')

FPS, FRAMES = 30.0, 31   # 1 second


def checkpoint_params(n=200, seed=1, sh_coeffs=8):
    g = np.random.default_rng(seed)
    f = lambda *s, sc=1.0: (g.standard_normal(s) * sc).astype(np.float32)  # noqa: E731
    q = f(n, 4)
    q /= np.linalg.norm(q, axis=1, keepdims=True)                   # xyzw, as the file stores them
    return {
        'means': f(n, 3), 'scales': f(n, 3, sc=0.1) - 3.0, 'quats': q, 'opacities': f(n), 'sh0': f(n, 1, 3, sc=0.2),
        'shN': f(n, sh_coeffs, 3, sc=0.05), 'times': g.uniform(0.0, 1.0, n).astype(np.float32),
        'log_durations': np.log(g.uniform(0.15, 0.6, n)).astype(np.float32), 'anchors': f(n, 3),
        'zeta_base_r': f(n, 3, sc=0.5), 'zeta_base_t': f(n, 3, sc=0.5),
        'zeta_c0_r': f(n, 3, sc=0.3), 'zeta_c0_t': f(n, 3, sc=0.3), 'zeta_c1_r': f(n, 3, sc=0.3), 'zeta_c1_t': f(n, 3, sc=0.3),
        'zeta_c2_r': f(n, 3, sc=0.3), 'zeta_c2_t': f(n, 3, sc=0.3)}


def pack(tmp_path, params, extra=()):
    ckpt = tmp_path / 'trigs_ckpt.pt'
    duration = (FRAMES - 1) / FPS
    torch.save({'params': {k: torch.from_numpy(v) for k, v in params.items()}, 'step': 100, 'fps': FPS, 'n_frames': FRAMES,
                'metrics': {'windows': [[0.0, duration]], 'ownership_bounds': [0.0, duration]}}, ckpt)
    out = tmp_path / 'final.trigsbin'
    done = subprocess.run([sys.executable, '-I', str(PACKER), 'from-checkpoint', '--checkpoint', str(ckpt), '--frame-rate', str(FPS),
                           '--output', str(out), '--overwrite', *extra], capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:] + done.stdout[-500:]
    return out


def reference(params):
    """The checkpoint itself, as TrigsTensors in wxyz (what the trainer sliced)."""
    t = lambda k: torch.from_numpy(params[k].astype(np.float32))  # noqa: E731
    q = t('quats')
    return trigs.TrigsTensors(
        means=t('means'), quats_wxyz=torch.cat([q[:, 3:4], q[:, :3]], 1), log_scales=t('scales'), opacity_logit=t('opacities'),
        sh=torch.cat([t('sh0'), t('shN')], 1), times=t('times'), log_sigma=t('log_durations'), anchors=t('anchors'),
        **{k: t(k) for k in ('zeta_base_r', 'zeta_base_t', 'zeta_c0_r', 'zeta_c0_t', 'zeta_c1_r', 'zeta_c1_t', 'zeta_c2_r', 'zeta_c2_t')})


def test_a_file_that_is_not_a_trigsbin_is_refused(tmp_path):
    bad = tmp_path / 'x.trigsbin'
    bad.write_bytes(b'NOTTRIGS' + b'\0' * 100)
    with pytest.raises(trigsbin.TrigsbinError, match='not a .trigsbin'):
        trigsbin.read(bad)
    short = tmp_path / 'y.trigsbin'
    short.write_bytes(b'TRIG')
    with pytest.raises(trigsbin.TrigsbinError):
        trigsbin.read(short)


@needs_packer
def test_a_packed_checkpoint_reads_back_with_every_tensor(tmp_path):
    params = checkpoint_params()
    tf = trigsbin.read(pack(tmp_path, params))
    assert tf.motion_model == 'trigs.se3_bezier_v1' and tf.fps == FPS
    assert len(tf.clips) == 1 and tf.count == 200
    got = tf.clips[0].tensors
    for k in ('means', 'quats', 'times', 'log_durations', 'anchors', 'zeta_base_r', 'zeta_c2_t'):   # motion32: float32, exact
        assert np.array_equal(got[k], params[k]), k
    for k in ('scales', 'opacities', 'sh0', 'shN'):                                                  # float16
        assert np.allclose(got[k], params[k], atol=2e-3, rtol=2e-3), k
    assert tf.clips[0].core == (0, FRAMES)


@needs_packer
@pytest.mark.parametrize('t', [0.0, 0.31, 0.5, 0.97])
def test_the_packed_model_slices_like_the_checkpoint(tmp_path, t):
    params = checkpoint_params()
    model = trigsbin.load_model(pack(tmp_path, params))
    means, quats, opac, sh, scales = model.slice(t)
    # every splat is kept (a one-window file culls nothing that is visible), so compare through the same ordering
    ref = reference(params)
    rm, rq, ro = trigs.slice_at(ref, t)
    assert means.shape == rm.shape
    # the packager keeps row order when nothing is culled
    assert torch.allclose(means, rm, atol=2e-3)
    assert torch.allclose(quats, rq, atol=2e-3)
    assert torch.allclose(opac, ro, atol=2e-3)
    assert torch.allclose(scales, ref.scales, atol=2e-3, rtol=2e-3)
    assert sh.shape == ref.sh.shape


ASOG_SRC = Path(os.environ.get('ASOG_ROOT') or Path.home() / 'Dev/github/animated-sog') / 'python' / 'src'
needs_asog = pytest.mark.skipif(not (ASOG_SRC / 'asog').is_dir(), reason='animated-sog is not checked out (set ASOG_ROOT)')


@needs_packer
@needs_asog
@pytest.mark.parametrize('t', [0.1, 0.5, 0.9])
def test_an_asog_decode_of_the_same_file_agrees_with_our_slice(tmp_path, t):
    """The animated-sog encoder and decoder (MIT, github.com/arrival-space/animated-sog) read the packer's file and evaluate it
    with their own code. The encoder reorders splats and quantises, so the comparison is of sorted weights and positions."""
    sys.path.insert(0, str(ASOG_SRC))
    try:
        import asog
    finally:
        sys.path.remove(str(ASOG_SRC))
    params = checkpoint_params(n=600)
    packed = pack(tmp_path, params)
    asog.encode(packed, tmp_path / 'c.asog', prune=0.0, shk=64)
    decoded = asog.read(tmp_path / 'c.asog')
    frame = decoded.evaluate(t)
    means, quats, opac, sh, scales = trigsbin.load_model(packed).slice(t)
    weight = (opac / torch.sigmoid(torch.from_numpy(params['opacities']))).numpy()
    assert decoded.count == len(weight)
    # the temporal weights: same formula, mu and sigma quantised (2^-11 s, 8-bit log sigma)
    assert np.allclose(np.sort(frame.weights), np.sort(weight), atol=2e-2)
    live_theirs, live_ours = frame.weights > 0.5, weight > 0.5
    assert abs(int(live_theirs.sum()) - int(live_ours.sum())) <= 6
    # where the splats are: every coordinate's sorted values (the motion maths, whatever order the file kept)
    # (quantiles: a splat on the weight threshold may count on one side only, so the two lists can differ by one)
    q = np.linspace(0.0, 1.0, 21)
    for axis in range(3):
        assert np.allclose(np.quantile(frame.positions[live_theirs, axis], q), np.quantile(means.numpy()[live_ours, axis], q), atol=0.05)


@needs_packer
def test_a_per_splat_time_slices_a_single_clip_file(tmp_path):
    params = checkpoint_params()
    model = trigsbin.load_model(pack(tmp_path, params))
    t = torch.full((200,), 0.4)
    means, *_ = model.slice(t)
    assert torch.allclose(means, model.slice(0.4)[0], atol=1e-6)

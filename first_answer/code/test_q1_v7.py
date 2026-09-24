import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from q1_v7_core import (Candidate, Quality, baseline, candidates, effective_n,
                        filter_extended, fit_operator, load_data, quality_features)


def example(n=40):
    rng = np.random.default_rng(12)
    time = np.arange(-51, 205) / 256
    x = baseline(rng.normal(size=(n, 3, 256)), time)
    q = quality_features(x)
    labels = np.tile([-1, 1], n // 2)
    meta = pd.DataFrame({"record_id": ["R"] * n, "trial_id": np.arange(1, n + 1),
                         "trial_key": [f"R-{i}" for i in range(n)], "cv_fold": np.arange(n) // 10})
    return x, q, labels, meta, time


def test_projection_and_baseline_are_preserved():
    x, q, y, meta, time = example()
    op = fit_operator(x[:30], q[:30], y[:30], meta.iloc[:30], time, Candidate("test", 4, 10., True))
    out = op.transform(x[30:])
    u = op.basis[:, :4]
    assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-12)
    assert_allclose(out.reshape(10, -1) @ u, x[30:].reshape(10, -1) @ u, atol=1e-10)
    assert_allclose(out[..., time < 0].mean(axis=-1), 0, atol=1e-10)
    assert_allclose(op.transform(x[30]), out[0], atol=1e-12)


def test_zero_strength_is_identity():
    x, q, y, meta, time = example()
    op = fit_operator(x, q, y, meta, time, Candidate("identity", 4, 0., False))
    assert_allclose(op.transform(x), x, atol=1e-12)


def test_fit_does_not_use_validation_values_or_labels():
    x, q, y, meta, time = example()
    c = Candidate("test", 4, 10., True)
    op1 = fit_operator(x[:30], q[:30], y[:30], meta.iloc[:30], time, c)
    x[30:] = 1e9
    y[30:] *= -1
    q[30:] = 1e9
    op2 = fit_operator(x[:30], q[:30], y[:30], meta.iloc[:30], time, c)
    assert_allclose(op1.basis, op2.basis, atol=0)
    assert_allclose(op1.quality.center, op2.quality.center, atol=0)
    assert set(op1.training_keys).isdisjoint(set(meta.iloc[30:].trial_key))


def test_filter_matches_readonly_v6_interface():
    data, meta = load_data()
    actual = filter_extended(data["extended"][:8], data["extended_time"], data["time"])
    assert_allclose(actual, data["filtered"][:8], atol=1e-10)
    assert meta.groupby("task").size().tolist() == [200, 200]
    assert len(candidates()) == 11


def test_quality_coordinate_and_effective_n():
    native = np.ones((4, 3, 256)) * 1000
    assert_allclose(quality_features(native)[:, 3], 1)
    assert_allclose(quality_features(native - 1000)[:, 3], 0)
    assert effective_n(np.ones(40)) == 40
    weights = Quality.fit(quality_features(native)).weights(quality_features(native))
    assert np.isfinite(weights).all() and np.all(weights > 0)


def test_linear_increment_is_consistent():
    x, q, y, meta, time = example()
    op = fit_operator(x, q, y, meta, time, Candidate("test", 2, 100., True))
    rng = np.random.default_rng(1)
    signal = baseline(rng.normal(size=x[:3].shape), time)
    assert_allclose(op.transform(x[:3] + signal) - op.transform(x[:3]),
                    op.transform(signal), atol=1e-10)


def test_contrasts_are_not_confused():
    from q1_v7_statistics import block_contrasts
    x, _, labels, meta, time = example()
    x[:] = 0
    for i, sign in enumerate(labels):
        x[i, :, time >= 0] = 3 if sign == -1 else 1
    meta["task"] = 1
    meta["cue_sign"] = labels
    blocks, missing = block_contrasts(x, meta, time, 10)
    assert not missing
    assert_allclose(blocks.common, 2)
    assert_allclose(blocks.difference, 2)
    assert_allclose(blocks.n_left+blocks.n_right, 10)


def test_holm_and_zero_null():
    from q1_v7_statistics import holm, sign_flip_test
    assert_allclose(holm([.04, .01, .03]), [.06, .03, .06])
    t, p = sign_flip_test(np.zeros(20), 999, np.random.default_rng(10))
    assert t == 0 and p == 1


def test_known_biphasic_curve_recovery():
    from q1_v7_fitting import evaluate, fit_curve
    time = np.arange(-51, 205)/256
    theta = np.array([.1, -.7, .12, .04, 1.2, .37, .07])
    truth = evaluate(theta, time)
    fitted = fit_curve(truth, time, 2, 1.)
    assert fitted.success
    assert_allclose(evaluate(fitted.theta, time[time >= 0]), truth[time >= 0], atol=1e-4)


def test_orthogonal_stress_is_not_a_protected_template():
    from q1_v7_injection import make_signals
    x, q, labels, meta, time = example()
    op = fit_operator(x, q, labels, meta, time, Candidate("test", 4, 10., True))
    ext_time = np.arange(-256, 384)/256
    waves = make_signals(ext_time, time, op.basis, 1.)
    stress = waves["outside_subspace_stress"][:, 205:461].ravel()
    assert_allclose(op.basis.T @ stress, 0, atol=1e-10)


def test_recovery_gate_uses_all_channels_not_only_F3():
    from q1_v7_injection import recovery_metrics
    time = np.arange(-51, 205)/256
    truth = np.ones((3, 1))*np.exp(-.5*((time-.35)/.05)**2)
    restored = truth.copy()
    restored[0] *= .7
    metrics = recovery_metrics(restored, truth, time)
    assert_allclose(metrics["peak_ratio_F3"], 1.)
    assert_allclose(metrics["amplitude_bias_abs"], .3)


def test_slow_branch_preserves_known_smooth_wave_not_proven_real_signal():
    ext_time = np.arange(-256, 384)/256
    time = np.arange(-51, 205)/256
    signal = np.ones((1, 3, 1))*np.exp(-.5*((ext_time-.42)/.22)**2)
    result = filter_extended(signal, ext_time, time, slow=True)
    truth = baseline(signal[..., 205:461], time)
    assert_allclose(result, truth, atol=1e-8)

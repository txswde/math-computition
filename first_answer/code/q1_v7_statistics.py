"""Prespecified contrasts and finite-budget stability checks, not population inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from q1_v7_core import (candidates, config, correlation, effective_n, fit_operator,
                        folder, load_data, mean_by_condition, save_json, save_table,
                        select_candidate)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.)
    return result


def sign_flip_test(values, repeats, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan, np.nan
    observed = values.mean() / max(values.std(ddof=1)/np.sqrt(len(values)), 1e-12)
    signs = rng.choice([-1., 1.], size=(repeats, len(values)))
    null = signs * values[None, :]
    t_null = null.mean(axis=1) / np.maximum(null.std(axis=1, ddof=1)/np.sqrt(len(values)), 1e-12)
    p = (1 + np.sum(np.abs(t_null) >= abs(observed)-1e-12)) / (repeats+1)
    return float(observed), float(p)


def moving_block_sample(indices, rng, span=2):
    indices = np.asarray(indices)
    if len(indices) == 0:
        return indices
    length = min(span, len(indices))
    starts = rng.integers(0, len(indices), size=int(np.ceil(len(indices)/length)))
    return np.concatenate([indices[(s+np.arange(length)) % len(indices)] for s in starts])[:len(indices)]


def bootstrap_trial_indices(meta, rng, block_size=10):
    result = []
    for _, group in meta.groupby("record_id", sort=True):
        keys = ((group.trial_id.to_numpy()-1)//block_size)
        blocks = np.unique(keys)
        for block in moving_block_sample(blocks, rng, 2):
            result.extend(group.index[keys == block].tolist())
    return np.asarray(result, dtype=int)


def block_contrasts(x, meta, time, block_size):
    pre = time < 0
    late = (time >= .25) & (time <= .5)
    amplitude = x[:, :, late].mean(axis=2)-x[:, :, pre].mean(axis=2)
    result, missing = [], []
    for record, group in meta.groupby("record_id", sort=True):
        block_ids = (group.trial_id - 1)//block_size
        for block in sorted(block_ids.unique()):
            rows = group.index[block_ids == block].to_numpy()
            signs = meta.loc[rows, "cue_sign"].to_numpy()
            if np.unique(signs).size != 2:
                missing.append({"record_id": record, "block": int(block), "n": len(rows)})
                continue
            left, right = amplitude[rows[signs == -1]].mean(axis=0), amplitude[rows[signs == 1]].mean(axis=0)
            for channel, name in enumerate(config()["channels"]):
                result.append({"task": int(group.task.iloc[0]), "record_id": record, "block": int(block),
                               "block_size": block_size, "channel": name,
                               "n_left": int((signs == -1).sum()), "n_right": int((signs == 1).sum()),
                               "left": left[channel], "right": right[channel],
                               "common": (left[channel]+right[channel])/2, "difference": left[channel]-right[channel]})
    return pd.DataFrame(result), missing


def run_statistics():
    cfg = config()
    data, meta = load_data()
    rng = np.random.default_rng(cfg["seed"] + 101)
    block_frames, summaries, missing_all, record_rows = [], [], [], []
    for block_size in cfg["stat_block_sizes"]:
        blocks, missing = block_contrasts(data["filtered"], meta, data["time"], block_size)
        missing_all.extend([{"block_size": block_size, **row} for row in missing])
        block_frames.append(blocks)
        local = []
        for (task, channel), group in blocks.groupby(["task", "channel"], sort=True):
            group = group.reset_index(drop=True)
            for contrast in ["common", "difference"]:
                values = group[contrast].to_numpy()
                stat, p = sign_flip_test(values, cfg["sign_flip_repeats"], rng)
                boot = []
                record_groups = [np.flatnonzero(group.record_id.to_numpy() == r) for r in group.record_id.unique()]
                for _ in range(cfg["bootstrap_repeats"]):
                    chosen = np.concatenate([moving_block_sample(indices, rng) for indices in record_groups])
                    boot.append(np.mean(values[chosen]))
                acfs, skews, trend_p = [], [], []
                for record, rec in group.groupby("record_id", sort=True):
                    v = rec[contrast].to_numpy()
                    acf = correlation(v[:-1], v[1:])
                    asymmetry = float(skew(v, bias=False)) if len(v) >= 3 else np.nan
                    trend = spearmanr(np.arange(len(v)), v)
                    acfs.append(abs(acf))
                    skews.append(abs(asymmetry))
                    trend_p.append(trend.pvalue)
                    record_rows.append({"task": task, "channel": channel, "contrast": contrast,
                                        "block_size": block_size, "record_id": record, "n_blocks": len(v),
                                        "effect": np.mean(v), "lag1_correlation": acf, "skewness": asymmetry,
                                        "time_trend_rho": trend.statistic, "time_trend_p_diagnostic": trend.pvalue})
                warning = (np.nanmax(acfs) > .4 or np.nanmax(skews) > 1. or np.nanmin(trend_p) < .05)
                local.append({"task": task, "channel": channel, "contrast": contrast,
                              "block_size": block_size, "n_blocks": len(values), "effect": values.mean(),
                              "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                              "t_stat": stat, "p_conditional": p, "assumption_warning": bool(warning),
                              "max_abs_lag1": np.nanmax(acfs), "max_abs_skew": np.nanmax(skews),
                              "inferential_status": "conditional_internal_not_population",
                              "n_trials_included": int((group.n_left+group.n_right).sum())})
        frame = pd.DataFrame(local)
        frame["p_holm_conditional"] = holm(frame.p_conditional.to_numpy())
        frame["primary_family"] = block_size == 10
        summaries.append(frame)
    summary = pd.concat(summaries, ignore_index=True)
    save_table("response_block_values.csv", pd.concat(block_frames, ignore_index=True))
    save_table("response_statistics.csv", summary)
    save_table("response_record_diagnostics.csv", record_rows)
    save_json("response_statistics_protocol.json", {
        "primary_family_n": 12, "primary_block_trials": 10, "two_sided": True,
        "window_s": [.25, .5], "missing_condition_blocks": missing_all,
        "primary_target": "fixed P0; no learned subspace or peak-window selection",
        "bootstrap_note": "500 circular moving-block resamples of adjacent block means within each record; pointwise not simultaneous intervals.",
        "assumption_note": "All p-values are conditional on independent sign-exchangeable block contrasts. Diagnostics cannot prove these assumptions.",
        "warning_rule": "Diagnostic only: any record abs(lag1)>0.4, abs(skew)>1, or Spearman trend p<0.05. Not a validated assumption-selection test.",
        "interpretation": "Average nonzero does not prove both classes respond, a P300 generator, or population generalization.",
    })
    print("Q1 primary response tests:", summary[summary.primary_family][
        ["task", "channel", "contrast", "effect", "p_holm_conditional", "assumption_warning"]].to_dict("records"), flush=True)


def run_split_half():
    cfg = config()
    data, meta = load_data()
    rng = np.random.default_rng(cfg["seed"] + 102)
    x, q, time = data["filtered"], data["quality"], data["time"]
    post = time >= 0
    choices = {c.name: c for c in candidates()}
    output, selection = [], []
    for task in [1, 2]:
        dev = meta[(meta.task == task) & (meta.split_role == "development")]
        for repeat in range(cfg["split_half_repeats"]):
            halves = [[], []]
            for _, rec in dev.groupby("record_id", sort=True):
                blocks = np.unique((rec.trial_id.to_numpy()-1)//10)
                rng.shuffle(blocks)
                for half in [0, 1]:
                    chosen = blocks[:len(blocks)//2] if half == 0 else blocks[len(blocks)//2:]
                    halves[half].extend(rec.index[np.isin((rec.trial_id.to_numpy()-1)//10, chosen)].tolist())
            curves = {m: [] for m in ["P0", "W0", "SELECTED"]}
            for half, rows in enumerate(halves):
                idx = np.asarray(rows)
                labels = meta.iloc[idx].cue_sign.to_numpy()
                selected, _ = select_candidate(x[idx], q[idx], labels, meta.iloc[idx], time)
                selection.append({"task": task, "repeat": repeat, "half": half, "selected": selected.name})
                for method in curves:
                    op = fit_operator(x[idx], q[idx], labels, meta.iloc[idx], time,
                                      selected if method == "SELECTED" else choices[method])
                    means = mean_by_condition(x[idx], labels, op.weights(q[idx]))
                    curves[method].append(op.transform(means))
            for method, (a, b) in curves.items():
                for contrast in ["common", "difference"]:
                    aa = a.mean(axis=0) if contrast == "common" else a[0]-a[1]
                    bb = b.mean(axis=0) if contrast == "common" else b[0]-b[1]
                    for e, channel in enumerate(cfg["channels"]):
                        output.append({"task": task, "repeat": repeat, "method": method,
                                       "contrast": contrast, "channel": channel,
                                       "correlation": correlation(aa[e, post], bb[e, post]),
                                       "rmse": np.sqrt(np.mean((aa[e, post]-bb[e, post])**2)),
                                       "rms_half_a": np.sqrt(np.mean(aa[e, post]**2)),
                                       "rms_half_b": np.sqrt(np.mean(bb[e, post]**2))})
        print(f"Q1 task={task} independent-half fits complete", flush=True)
    save_table("split_half_metrics.csv", output)
    save_table("split_half_selection.csv", selection)


def run_erp_curves():
    cfg = config()
    data, meta = load_data()
    rng = np.random.default_rng(cfg["seed"] + 103)
    with np.load(folder() / "cleaned_epochs.npz") as archive:
        processed = {key: archive[key] for key in archive.files}
    time = data["time"]
    rows, counts = [], []
    for task in [1, 2]:
        idx = np.flatnonzero(meta.task.to_numpy() == task)
        local_meta = meta.iloc[idx].reset_index(drop=True)
        labels = local_meta.cue_sign.to_numpy()
        bootstrap = [bootstrap_trial_indices(local_meta, rng) for _ in range(cfg["bootstrap_repeats"])]
        for method in ["P0", "W0", "SELECTED"]:
            x = processed[f"{method}_epochs"][idx]
            w = processed[f"{method}_weights"][idx]
            for sign in [-1, 1]:
                chosen = labels == sign
                mean = np.average(x[chosen], axis=0, weights=w[chosen])
                boot = []
                for sampled in bootstrap:
                    sampled = sampled[labels[sampled] == sign]
                    boot.append(np.average(x[sampled], axis=0, weights=w[sampled]))
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                counts.append({"task": task, "condition": sign, "method": method,
                               "n": int(chosen.sum()), "effective_n": effective_n(w[chosen]),
                               "scope": "descriptive_all_crossfit", "coverage": 1.})
                for e, channel in enumerate(cfg["channels"]):
                    for t, value in enumerate(time):
                        rows.append({"task": task, "condition": sign, "channel": channel, "method": method,
                                     "time_s": value, "mean": mean[e, t], "ci_low": lo[e, t], "ci_high": hi[e, t]})
    save_table("erp_curves.csv", rows)
    save_table("erp_sample_counts.csv", counts)
    save_json("erp_interval_note.json", {"scope": "Descriptive 400-trial crossfit means.",
                                       "interval": "Pointwise conditional on already fitted fold operators and weights; not full-pipeline confidence bands.",
                                       "unit_of_resampling": "record-local consecutive 10-trial blocks, span two blocks"})


def run_sensitivity():
    """Prespecified filter/padding branches; never select by their p-values."""
    from q1_v7_core import baseline, filter_extended
    from q1_v7_injection import make_signals, recovery_metrics
    cfg = config()
    data, meta = load_data()
    with np.load(folder() / "baseline_branches.npz") as z:
        branches = {"P0": data["filtered"], "SLOW30": z["slow"],
                    "P0_LARGER_PADDING": z["padding_sensitivity"]}
    summaries, curves, fidelity = [], [], []
    for name, x in branches.items():
        # Common random numbers isolate changes caused by preprocessing.
        rng = np.random.default_rng(cfg["seed"]+105)
        blocks, missing = block_contrasts(x, meta, data["time"], 10)
        assert not missing
        local = []
        for (task, channel), group in blocks.groupby(["task", "channel"], sort=True):
            group = group.reset_index(drop=True)
            rec_indices = [np.flatnonzero(group.record_id.to_numpy() == r) for r in group.record_id.unique()]
            for contrast in ["common", "difference"]:
                values = group[contrast].to_numpy()
                stat, p = sign_flip_test(values, cfg["sign_flip_repeats"], rng)
                boot = [np.mean(values[np.concatenate([moving_block_sample(i, rng) for i in rec_indices])])
                        for _ in range(cfg["bootstrap_repeats"])]
                local.append({"branch": name, "task": task, "channel": channel, "contrast": contrast,
                              "effect": values.mean(), "ci_low": np.percentile(boot, 2.5),
                              "ci_high": np.percentile(boot, 97.5), "p_conditional": p,
                              "role": "secondary_sensitivity_not_new_confirmation"})
        frame = pd.DataFrame(local)
        frame["p_holm_within_branch"] = holm(frame.p_conditional.to_numpy())
        summaries.append(frame)
        for task in [1, 2]:
            means = [x[(meta.task.to_numpy() == task) & (meta.cue_sign.to_numpy() == s)].mean(axis=0)
                     for s in [-1, 1]]
            for e, channel in enumerate(cfg["channels"]):
                for i, t in enumerate(data["time"]):
                    curves.append({"branch": name, "task": task, "channel": channel, "time_s": t,
                                   "common": (means[0][e, i]+means[1][e, i])/2,
                                   "difference": means[0][e, i]-means[1][e, i]})
    start = int(round((data["time"][0]-data["extended_time"][0])*cfg["fs"]))
    families = make_signals(data["extended_time"], data["time"], np.zeros((768, 0)), 1.)
    for name, slow in [("P0", False), ("SLOW30", True)]:
        for family, signal in families.items():
            if "subspace_stress" in family:
                continue
            truth = baseline(signal[:, start:start+256], data["time"])
            restored = filter_extended(signal[None], data["extended_time"], data["time"], slow=slow)[0]
            fidelity.append({"branch": name, "family": family, **recovery_metrics(restored, truth, data["time"])})
    save_table("branch_response_sensitivity.csv", pd.concat(summaries, ignore_index=True))
    save_table("branch_mean_curves.csv", curves)
    save_table("filter_fidelity_sensitivity.csv", fidelity)
    save_json("branch_sensitivity_note.json", {
        "slow": "30 Hz lowpass plus baseline; deliberately no fitted drift regression, so residual slow drift remains a confound.",
        "padding": "Same 0.5-30 Hz P0 filter on larger disjoint raw trial partitions; only boundary context changes.",
        "p_values": "Secondary conditional tests, Holm within each 12-test branch only; do not select the branch with the lowest p.",
        "common_random_numbers": "P0 is repeated with the same sensitivity seed as other branches; small Monte Carlo p variation vs primary is expected.",
    })

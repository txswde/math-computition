"""Numerical Q1 stages; nothing in the v6 output directory is written."""
from __future__ import annotations

import importlib
import json
import platform
import time as clock

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from q1_v7_core import (Candidate, Operator, baseline, candidates, config, correlation,
                        effective_n, filter_extended, fit_operator, folder, load_data,
                        mean_by_condition, protected_paths, quality_features,
                        save_json, save_table, select_candidate)
from utils import PROJECT_ROOT, load_eeg_mat, sha256_file


def e0():
    start = clock.perf_counter()
    before = protected_paths()
    snapshot = folder() / "protected_before.json"
    if snapshot.exists():
        saved = json.loads(snapshot.read_text(encoding="utf-8"))
        assert saved == before, "Historical protected files changed since the initial E0"
    else:
        save_json("protected_before.json", before)
    cfg = config()
    save_json("configuration.json", cfg)
    data, meta = load_data()
    reconstructed = filter_extended(data["extended"], data["extended_time"], data["time"])
    assert np.max(np.abs(reconstructed - data["filtered"])) < 1e-9
    assert np.max(np.abs(quality_features(data["native"]) - data["quality"])) < 1e-9
    raw_records = {r["record_id"]: r for r in
                   (load_eeg_mat(p) for p in (PROJECT_ROOT / "中文题目/C题").glob("*.mat"))}
    padding_rows, padded_curves = [], []
    sos = butter(4, [.5, 30.], fs=256, btype="bandpass", output="sos")
    scale = np.std(data["filtered"][meta.split_role.to_numpy() == "development"], axis=(0, 2))
    for record, group in meta.groupby("record_id", sort=True):
        mat = raw_records[record]
        assert mat["sample_rate"] == 256
        events = group.cue_start_idx.to_numpy(dtype=int)
        assert np.all(mat["data"][7, events] == group.cue_sign.to_numpy())
        for position, row in enumerate(group.itertuples()):
            event = int(row.cue_start_idx)
            # No raw samples are shared between neighboring trial partitions.
            left_boundary = 0 if position == 0 else int((events[position-1] + event) // 2)
            right_boundary = mat["data"].shape[1] if position == len(events)-1 else int((event + events[position+1]) // 2)
            left, right = max(event-512, left_boundary), min(event+768, right_boundary)
            segment = sosfiltfilt(sos, mat["data"][:3, left:right], axis=-1)
            curve = baseline(segment[:, event-left-51:event-left+205], data["time"])
            original = data["filtered"][row.row_id]
            padded_curves.append((row.row_id, curve))
            for channel, name in enumerate(cfg["channels"]):
                post = data["time"] >= 0
                padding_rows.append({"trial_key": row.trial_key, "task": row.task, "channel": name,
                                     "effective_pre_s": (event-left)/256, "effective_post_s": (right-event)/256,
                                     "rmse_native": np.sqrt(np.mean((curve[channel, post]-original[channel, post])**2)),
                                     "training_scaled_rmse": np.sqrt(np.mean((curve[channel, post]-original[channel, post])**2))/scale[channel]})
    padded = np.asarray([x[1] for x in sorted(padded_curves, key=lambda x: x[0])])
    slow = filter_extended(data["extended"], data["extended_time"], data["time"], slow=True)
    np.savez_compressed(folder() / "baseline_branches.npz", slow=slow, padding_sensitivity=padded,
                        time=data["time"])
    save_table("padding_sensitivity.csv", padding_rows)
    inventory = {"trials": len(meta), "development_trials": int((meta.split_role == "development").sum()),
                 "historical_test_trials": int((meta.split_role == "test").sum()),
                 "channels": cfg["channels"], "unit": "recording units; physical conversion unavailable",
                 "fs_hz": 256, "v6_filter_reproduction_max_abs_error": np.max(np.abs(reconstructed-data["filtered"])),
                 "quality_coordinates_match_v6": True, "candidate_count": len(candidates()),
                 "raw_and_historical_file_count": len(before), "unavailable_correctness_is_not_imputed": True,
                 "padding_note": "P0 retained as frozen historical baseline; larger disjoint raw partitions are a sensitivity branch, not chosen by significance.",
                 "python": platform.python_version(),
                 "packages": {m: importlib.import_module(m).__version__ for m in ["numpy", "scipy", "pandas", "matplotlib"]},
                 "elapsed_s": clock.perf_counter()-start}
    save_json("e0_inventory.json", inventory)
    print(json.dumps({"stage": "E0", "trials": 400, "candidates": len(candidates()),
                      "protected_files": len(before), "elapsed_s": inventory["elapsed_s"]}), flush=True)


def denoise():
    cfg = config()
    data, meta = load_data()
    x, q, time = data["filtered"], data["quality"], data["time"]
    labels = meta.cue_sign.to_numpy(dtype=int)
    methods = ["P0", "W0", "SELECTED"]
    cleaned = {name: np.zeros_like(x) for name in methods}
    weights = {name: np.ones(len(x)) for name in methods}
    choices = {c.name: c for c in candidates()}
    decisions, inner_rows, prediction_rows, trial_rows = [], [], [], []
    audits, model_arrays = {}, {}
    for task in [1, 2]:
        dev = np.flatnonzero((meta.task.to_numpy() == task) & (meta.split_role.to_numpy() == "development"))
        hist = np.flatnonzero((meta.task.to_numpy() == task) & (meta.split_role.to_numpy() == "test"))
        partitions = []
        for fold in sorted(meta.iloc[dev].cv_fold.unique()):
            partitions.append((f"outer_{int(fold)}", dev[meta.iloc[dev].cv_fold.to_numpy() != fold],
                               dev[meta.iloc[dev].cv_fold.to_numpy() == fold]))
        partitions.append(("historical", dev, hist))
        for partition, train, test in partitions:
            selected, detail = select_candidate(x[train], q[train], labels[train], meta.iloc[train], time)
            for row in detail:
                inner_rows.append({"task": task, "partition": partition, **row})
            decisions.append({"task": task, "partition": partition, "selected": selected.name,
                              "rank": selected.rank, "lambda": selected.strength,
                              "n_train": len(train), "n_test": len(test)})
            scale = np.maximum(np.std(x[train], axis=(0, 2)), 1e-6)
            for method in methods:
                candidate = selected if method == "SELECTED" else choices[method]
                op = fit_operator(x[train], q[train], labels[train], meta.iloc[train], time, candidate)
                assert set(meta.iloc[test].trial_key).isdisjoint(op.training_keys)
                cleaned[method][test] = op.transform(x[test])
                weights[method][test] = op.weights(q[test])
                trained = mean_by_condition(x[train], labels[train], op.weights(q[train]))
                prediction = op.transform(trained)
                truth = mean_by_condition(x[test], labels[test], np.ones(len(test)))
                post = time >= 0
                for sign_i, sign in enumerate([-1, 1]):
                    for e, channel in enumerate(cfg["channels"]):
                        predicted = prediction[sign_i, e, post]
                        observed = truth[sign_i, e, post]
                        prediction_rows.append({"task": task, "partition": partition, "method": method,
                                                "condition": sign, "channel": channel,
                                                "rmse": np.sqrt(np.mean((predicted-observed)**2)),
                                                "scaled_rmse": np.sqrt(np.mean((predicted-observed)**2))/scale[e],
                                                "correlation": correlation(predicted, observed),
                                                "prediction_norm": np.linalg.norm(predicted),
                                                "target_norm": np.linalg.norm(observed)})
                audits[f"{task}:{partition}:{method}"] = op.audit()
                if partition == "historical":
                    model_arrays[f"task{task}_{method}_U"] = op.basis
                    model_arrays[f"task{task}_{method}_quality_center"] = op.quality.center
                    model_arrays[f"task{task}_{method}_quality_spread"] = op.quality.spread
            print(f"Q1 task={task} partition={partition} selected={selected.name}", flush=True)
        # Per-trial data are out-of-fold for development and dev-fitted for historical test.
    for i, row in enumerate(meta.itertuples()):
        for method in methods:
            u = cleaned[method][i]
            trial_rows.append({"trial_key": row.trial_key, "task": row.task, "record_id": row.record_id,
                               "split_role": row.split_role, "method": method, "weight": weights[method][i],
                               "peak_to_peak_proxy": np.ptp(u, axis=-1).max(),
                               "first_difference_rms_proxy": np.sqrt(np.mean(np.diff(u, axis=-1)**2)),
                               "prebaseline_abs_error": np.abs(u[:, time < 0].mean(axis=-1)).max(),
                               "native_anomaly_fraction": q[i, 3], "retained": True})
    np.savez_compressed(folder() / "cleaned_epochs.npz", time=time,
                        **{f"{m}_epochs": v for m, v in cleaned.items()},
                        **{f"{m}_weights": v for m, v in weights.items()})
    np.savez_compressed(folder() / "final_operators.npz", **model_arrays)
    save_table("selection_decisions.csv", decisions)
    save_table("selection_inner_losses.csv", inner_rows)
    save_table("heldout_erp_prediction.csv", prediction_rows)
    save_table("trial_processing.csv", trial_rows)
    save_json("operator_fit_audit.json", audits)
    save_table("trial_metadata.csv", meta)


def check_protected():
    before = json.loads((folder() / "protected_before.json").read_text(encoding="utf-8"))
    after = protected_paths()
    assert before == after, "A protected historical/input file changed"
    return {"protected_files": len(after), "all_unchanged": True}

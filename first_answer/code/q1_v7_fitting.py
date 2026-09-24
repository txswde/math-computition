"""Finite Gaussian curve fitting with development-only complexity choice."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from q1_v7_core import (candidates, config, fit_operator, folder, load_data,
                        save_json, save_table)
from q1_v7_statistics import bootstrap_trial_indices


def evaluate(theta, time):
    result = np.ones_like(time, dtype=float) * theta[0]
    for offset in range(1, len(theta), 3):
        amplitude, mu, sigma = theta[offset:offset+3]
        result += amplitude*np.exp(-.5*((time-mu)/sigma)**2)
    return result


def parameter_bounds(k):
    lower, upper = [-6.], [6.]
    windows = [] if k == 0 else ([config()["late_window_s"]] if k == 1 else
                                [config()["early_window_s"], config()["late_window_s"]])
    for lo, hi in windows:
        lower.extend([-6., lo, .015])
        upper.extend([6., hi, .20])
    return np.asarray(lower), np.asarray(upper)


@dataclass
class CurveFit:
    theta: np.ndarray
    success: bool
    nfev: int
    cost: float
    scale: float
    boundary: bool


def fit_curve(y, time, k, scale, warm=None):
    cfg = config()
    score = (time >= 0) & (time < .8)
    scale = max(float(scale), 1e-6)
    target = y[score]/scale
    t = time[score]
    if k == 0:
        theta = np.array([np.mean(target)])
        return CurveFit(theta, True, 1, float(np.sum((target-theta[0])**2)/2), scale, False)
    lower, upper = parameter_bounds(k)
    starts = []
    if warm is not None:
        starts = [np.clip(warm, lower+1e-8, upper-1e-8)]
    else:
        for start in range(cfg["gaussian_starts"]):
            b = float(np.clip(np.median(target[:13]), -5., 5.))
            theta = [b]
            windows = [cfg["late_window_s"]] if k == 1 else [cfg["early_window_s"], cfg["late_window_s"]]
            for lo, hi in windows:
                mu = lo+(hi-lo)*(start+.5)/cfg["gaussian_starts"]
                amplitude = np.interp(mu, t, target)-b
                theta.extend([float(np.clip(amplitude, -5., 5.)), mu, .055+.01*start])
            starts.append(np.asarray(theta))
    results = []
    for theta in starts:
        result = least_squares(lambda p: evaluate(p, t)-target, theta, bounds=(lower, upper),
                               max_nfev=cfg["gaussian_max_nfev"], ftol=cfg["gaussian_tolerance"],
                               xtol=cfg["gaussian_tolerance"], gtol=cfg["gaussian_tolerance"])
        results.append(result)
    valid = [r for r in results if r.success] or results
    best = min(valid, key=lambda r: r.cost)
    boundary = bool(np.any(np.minimum((best.x-lower)/(upper-lower),
                                     (upper-best.x)/(upper-lower)) < 1e-3))
    return CurveFit(best.x, bool(best.success), int(best.nfev), float(best.cost), scale, boundary)


def run():
    cfg = config()
    data, meta = load_data()
    time = data["time"]
    x, q = data["filtered"], data["quality"]
    gate = json.loads((folder() / "q1_interface_gate.json").read_text(encoding="utf-8"))
    chosen_methods = {g["task"]: g["provisional_downstream_interface"] for g in gate}
    choices = {c.name: c for c in candidates()}
    rng = np.random.default_rng(cfg["seed"] + 104)
    parameter_rows, fit_rows, cv_rows, bootstrap_rows, fit_models = [], [], [], [], {}
    accepted = np.zeros_like(x)
    accepted_weights = np.ones(len(x))
    for task in [1, 2]:
        dev = np.flatnonzero((meta.task.to_numpy() == task) & (meta.split_role.to_numpy() == "development"))
        hist = np.flatnonzero((meta.task.to_numpy() == task) & (meta.split_role.to_numpy() == "test"))
        local_meta = meta.iloc[dev].reset_index(drop=True)
        labels = local_meta.cue_sign.to_numpy()
        method = chosen_methods[task]
        op = fit_operator(x[dev], q[dev], labels, local_meta, time, choices[method])
        train_clean = op.transform(x[dev])
        train_weights = op.weights(q[dev])
        # Final development-fitted arrays are for descriptive use, not OOF scores.
        all_task = np.r_[dev, hist]
        accepted[all_task] = op.transform(x[all_task])
        accepted_weights[all_task] = op.weights(q[all_task])
        scale = np.std(x[dev], axis=(0, 2))
        fold_cache = []
        for fold in sorted(local_meta.cv_fold.unique()):
            ti = np.flatnonzero(local_meta.cv_fold.to_numpy() != fold)
            vi = np.flatnonzero(local_meta.cv_fold.to_numpy() == fold)
            fold_op = fit_operator(x[dev[ti]], q[dev[ti]], labels[ti], local_meta.iloc[ti], time, choices[method])
            fold_cache.append((fold, ti, vi, fold_op.transform(x[dev[ti]]), fold_op.weights(q[dev[ti]])))
        sampled_blocks = [bootstrap_trial_indices(local_meta, rng) for _ in range(cfg["parameter_bootstrap_repeats"])]
        for sign in [-1, 1]:
            mask = labels == sign
            mean = np.average(train_clean[mask], axis=0, weights=train_weights[mask])
            test_mask = meta.iloc[hist].cue_sign.to_numpy() == sign
            historical_mean = x[hist[test_mask]].mean(axis=0)
            for e, channel in enumerate(cfg["channels"]):
                losses = {}
                for k in cfg["curve_complexity"]:
                    losses[k] = []
                    for fold, ti, vi, transformed, w in fold_cache:
                        if not np.any(labels[ti] == sign) or not np.any(labels[vi] == sign):
                            continue
                        training_mean = np.average(transformed[labels[ti] == sign, e], axis=0,
                                                   weights=w[labels[ti] == sign])
                        validation_mean = x[dev[vi[labels[vi] == sign]], e].mean(axis=0)
                        fit = fit_curve(training_mean, time, k, scale[e])
                        post = time >= 0
                        loss = np.mean((evaluate(fit.theta, time[post])*fit.scale-validation_mean[post])**2)/scale[e]**2
                        losses[k].append(loss)
                        cv_rows.append({"task": task, "condition": sign, "channel": channel, "k": k,
                                        "fold": fold, "method": method, "loss": loss,
                                        "converged": fit.success, "boundary": fit.boundary})
                selected_k = min(cfg["curve_complexity"], key=lambda k: np.mean(losses[k]) if losses[k] else np.inf)
                fitted = fit_curve(mean[e], time, selected_k, scale[e])
                boot_params, boot_curves = [], []
                for number, sampled in enumerate(sampled_blocks):
                    sampled = sampled[labels[sampled] == sign]
                    if len(sampled) < 2:
                        continue
                    boot_mean = np.average(train_clean[sampled, e], axis=0, weights=train_weights[sampled])
                    boot_fit = fit_curve(boot_mean, time, selected_k, scale[e], warm=fitted.theta)
                    bootstrap_rows.append({"task": task, "condition": sign, "channel": channel,
                                           "replicate": number, "converged": boot_fit.success,
                                           "boundary": boot_fit.boundary, "k": selected_k})
                    if boot_fit.success:
                        boot_params.append(boot_fit.theta)
                        boot_curves.append(evaluate(boot_fit.theta, time)*scale[e])
                p_low, p_high = np.percentile(boot_params, [2.5, 97.5], axis=0)
                curve_low, curve_high = np.percentile(boot_curves, [2.5, 97.5], axis=0)
                prediction = evaluate(fitted.theta, time)*scale[e]
                post = time >= 0
                common = {"task": task, "condition": sign, "channel": channel, "method": method,
                          "selected_k": selected_k, "n_train_trials": int(mask.sum()),
                          "n_historical_trials": int(test_mask.sum()),
                          "train_rmse": np.sqrt(np.mean((prediction[post]-mean[e, post])**2)),
                          "historical_P0_rmse": np.sqrt(np.mean((prediction[post]-historical_mean[e, post])**2)),
                          "converged": fitted.success, "boundary": fitted.boundary,
                          "best_start_nfev": fitted.nfev, "successful_bootstraps": len(boot_params),
                          "development_selection_loss": np.mean(losses[selected_k])}
                names = ["offset"]
                for component in (["late"] if selected_k == 1 else ["early", "late"] if selected_k == 2 else []):
                    names.extend([f"{component}_amplitude", f"{component}_mu_s", f"{component}_sigma_s"])
                for j, name in enumerate(names):
                    factor = scale[e] if name == "offset" or name.endswith("amplitude") else 1.
                    parameter_rows.append({**common, "parameter": name, "value": fitted.theta[j]*factor,
                                           "ci_low": p_low[j]*factor, "ci_high": p_high[j]*factor,
                                           "interval_scope": "conditional_on_operator_and_selected_K"})
                for k, instant in enumerate(time):
                    fit_rows.append({"task": task, "condition": sign, "channel": channel, "method": method,
                                     "selected_k": selected_k, "time_s": instant,
                                     "training_clean_mean": mean[e, k], "historical_P0_mean": historical_mean[e, k],
                                     "fit": prediction[k] if instant >= 0 else np.nan,
                                     "fit_ci_low": curve_low[k] if instant >= 0 else np.nan,
                                     "fit_ci_high": curve_high[k] if instant >= 0 else np.nan})
                fit_models[f"{task}:{sign}:{channel}"] = {**common, "theta_scaled": fitted.theta, "scale": scale[e]}
                print(f"Q1 curve task={task} shape={sign} {channel}: K={selected_k}, boundary={fitted.boundary}", flush=True)
    save_table("gaussian_parameters.csv", parameter_rows)
    save_table("gaussian_fitted_curves.csv", fit_rows)
    save_table("gaussian_complexity_selection.csv", cv_rows)
    save_table("gaussian_bootstrap_status.csv", bootstrap_rows)
    save_json("gaussian_models.json", fit_models)
    np.savez_compressed(folder() / "q1_downstream_interface.npz", signal=accepted, weights=accepted_weights,
                        time=time, trial_key=meta.trial_key.to_numpy(dtype=str))
    save_json("q1_downstream_interface_note.json", {
        "methods": chosen_methods, "development_scope": "fitted on all development data, descriptive only, not OOF",
        "historical_scope": "parameters trained only on original development; historical scores already inspected in prior versions",
        "for_Q2_Q3": "Use P0 raw/cache and refit all learned Q1 steps inside downstream training folds. Do not use this all-development fitted archive as an independent OOF benchmark.",
        "gaussian_ci": "100 block bootstraps with fixed operator, weights and chosen K; one warm start per replicate; conditional intervals, not full selection uncertainty.",
        "gaussian_boundary": "A boundary or amplitude interval spanning zero prevents a precise physiological peak interpretation.",
    })

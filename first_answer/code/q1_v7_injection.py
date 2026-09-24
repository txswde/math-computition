"""Known-signal perturbation tests: truth is pre-filter signal, not a fitted ERP."""
from __future__ import annotations

import numpy as np
import pandas as pd

from q1_v7_core import (baseline, candidates, config, effective_n, filter_extended,
                        fit_operator, folder, load_data, quality_features,
                        save_json, save_table)


def make_signals(time, crop_time, reference_basis, scale):
    def g(mu, sigma):
        return np.exp(-.5*((time-mu)/sigma)**2)
    specifications = [
        ("broad_common", g(.35, .08), [1., .8, .8]),
        ("narrow_antisym", g(.30, .025), [.2, 1., -1.]),
        ("shifted_antisym", g(.55, .045), [.2, 1., -1.]),
        ("biphasic_common", -.65*g(.16, .035)+g(.37, .055), [1., .8, .8]),
        ("oscillatory", np.cos(2*np.pi*9*(time-.35))*g(.35, .10), [.5, 1., -.6]),
        ("slow_common", g(.42, .22), [1., .8, .8]),
    ]
    output = {name: scale*np.asarray(spatial)[:, None]*wave[None, :] for name, wave, spatial in specifications}
    start = int(round((crop_time[0]-time[0])*256))
    template = baseline(output["broad_common"][:, start:start+len(crop_time)], crop_time).ravel()
    projection = reference_basis @ (reference_basis.T @ template)
    outside = template-projection
    if np.linalg.norm(outside) < 1e-8:
        raise ValueError("Degenerate orthogonal stress signal")
    for name, value in [("outside_subspace_stress", outside), ("mixed_subspace_stress", outside+.5*projection)]:
        value = value.reshape(3, len(crop_time))
        value *= scale/max(np.max(np.abs(value)), 1e-12)
        extended = np.zeros((3, len(time)))
        extended[:, start:start+len(crop_time)] = value
        output[name] = extended
    return output


def recovery_metrics(recovered, truth, time):
    post = time >= 0
    e = 1  # F3 has nonzero amplitude in every prespecified test family.
    exact = truth[e, post]
    restored = recovered[e, post]
    peak = int(np.argmax(np.abs(exact)))
    restored_peak = int(np.argmax(np.abs(restored)))
    true_peak = exact[peak]
    gain = restored[restored_peak]*np.sign(true_peak)/max(abs(true_peak), 1e-12)
    amplitude_errors, shifts = [], []
    for channel in range(3):
        original = truth[channel, post]
        output = recovered[channel, post]
        original_peak = int(np.argmax(np.abs(original)))
        output_peak = int(np.argmax(np.abs(output)))
        if abs(original[original_peak]) > 1e-8:
            channel_gain = output[output_peak]*np.sign(original[original_peak])/abs(original[original_peak])
            amplitude_errors.append(abs(channel_gain-1))
            shifts.append(abs(output_peak-original_peak))
    return {"peak_ratio_F3": float(gain),
            "amplitude_bias_abs": float(max(amplitude_errors)),
            "peak_shift_samples_F3": restored_peak-peak,
            "peak_shift_samples_max": int(max(shifts)),
            "relative_waveform_error": float(np.linalg.norm(recovered-truth)/max(np.linalg.norm(truth), 1e-12))}


def run():
    cfg = config()
    data, meta = load_data()
    choices = candidates()
    selected = pd.read_csv(folder() / "selection_decisions.csv")
    time, extended_time = data["time"], data["extended_time"]
    crop_start = int(round((time[0]-extended_time[0])*256))
    crop = slice(crop_start, crop_start+len(time))
    rows, examples, gates = [], [], []
    for task in [1, 2]:
        dev = np.flatnonzero((meta.task.to_numpy() == task) & (meta.split_role.to_numpy() == "development"))
        test = np.concatenate([group.index.to_numpy()[:10] for _, group in
                               meta[(meta.task == task) & (meta.split_role == "test")].groupby("record_id", sort=True)])
        labels = meta.iloc[dev].cue_sign.to_numpy()
        name = selected[(selected.task == task) & (selected.partition == "historical")].selected.iloc[0]
        operators = {c.name: fit_operator(data["filtered"][dev], data["quality"][dev], labels,
                                         meta.iloc[dev], time, c) for c in choices}
        reference = operators[name].basis
        scale = float(np.median(np.std(data["filtered"][dev], axis=-1)))
        signals = make_signals(extended_time, time, reference, scale)
        pulse = np.exp(-.5*((extended_time-.40)/.10)**2)
        artifacts = {
            "blink_like": 3*scale*np.array([1., .6, .6])[:, None]*pulse[None],
            "muscle_like": 3*scale*np.array([.3, 1., .5])[:, None]*
                           (np.cos(2*np.pi*45*extended_time)*pulse)[None],
            "step_like": 3*scale*np.ones((3, 1))*((extended_time >= .28) & (extended_time < .56))[None],
        }
        x = data["extended"][test]
        contaminated = (np.arange(len(x)) % 2 == 0).astype(float)[:, None, None]
        filtered_background = data["filtered"][test]
        for c in choices:
            op = operators[c.name]
            def pipeline(ext):
                transformed = op.transform(filter_extended(ext, extended_time, time))
                weights = op.weights(quality_features(ext[:, :, crop]))
                return np.average(transformed, axis=0, weights=weights), effective_n(weights)
            background_mean, _ = pipeline(x)
            for family, signal in signals.items():
                truth = baseline(signal[:, crop], time)
                filtered_signal = filter_extended(signal[None], extended_time, time)[0]
                restored = op.transform(filtered_signal)
                # Verify the actual raw-entry increment rather than relying only on algebra.
                increments = op.transform(filter_extended(x[:2]+signal, extended_time, time))-op.transform(filtered_background[:2])
                linearity_error = float(np.max(np.abs(increments-restored)))
                if linearity_error > 1e-8:
                    raise AssertionError("Raw-entry signal increment mismatch")
                fixed_metrics = recovery_metrics(restored, truth, time)
                unit = truth.ravel()
                u = op.basis[:, :c.rank]
                fraction_out = np.linalg.norm(unit-u@(u.T@unit))/max(np.linalg.norm(unit), 1e-12)
                rows.append({"task": task, "method": c.name, "is_selected": c.name == name,
                             "family": family, "scope": "single_trial_operator", "artifact": "none",
                             "artifact_removal_fraction": np.nan, "effective_n": len(test),
                             "out_of_candidate_subspace_fraction": fraction_out,
                             "linearity_max_abs_error": linearity_error,
                             "incremental_error_vs_P0": np.linalg.norm(restored-filtered_signal)/max(np.linalg.norm(filtered_signal), 1e-12),
                             **fixed_metrics})
                mean_signal, neff = pipeline(x+signal)
                for artifact_name, artifact in artifacts.items():
                    full, full_neff = pipeline(x+signal+contaminated*artifact)
                    only_artifact, _ = pipeline(x+contaminated*artifact)
                    residual = full-mean_signal
                    original_artifact = baseline((contaminated*artifact).mean(axis=0)[:, crop], time)
                    removal = 1-np.sum(residual**2)/max(np.sum(original_artifact**2), 1e-12)
                    rows.append({"task": task, "method": c.name, "is_selected": c.name == name,
                                 "family": family, "scope": "weighted_ERP_pipeline", "artifact": artifact_name,
                                 "artifact_removal_fraction": removal, "effective_n": full_neff,
                                 "out_of_candidate_subspace_fraction": fraction_out,
                                 "linearity_max_abs_error": np.nan, "incremental_error_vs_P0": np.nan,
                                 **recovery_metrics(full-only_artifact, truth, time)})
                if c.name in {"P0", name}:
                    for k, instant in enumerate(time):
                        examples.append({"task": task, "method": c.name, "family": family, "time_s": instant,
                                         "truth_F3": truth[1, k], "restored_F3": restored[1, k]})
        print(f"Q1 task={task} raw-entry recovery and artifact tradeoffs complete", flush=True)
    frame = save_table("injection_metrics.csv", rows)
    save_table("injection_curves.csv", examples)
    for task in [1, 2]:
        subset = frame[(frame.task == task) & (frame.scope == "single_trial_operator")]
        name = selected[(selected.task == task) & (selected.partition == "historical")].selected.iloc[0]
        primary = subset[(subset.method == name) & subset.family.isin(["broad_common", "narrow_antisym"])]
        passes = bool((primary.amplitude_bias_abs <= cfg["injection_amplitude_tolerance"]).all()
                      and (primary.peak_shift_samples_max <= cfg["injection_peak_tolerance_samples"]).all())
        gates.append({"task": task, "selected_by_training_prediction": name,
                      "broad_narrow_absolute_fidelity_pass": passes,
                      "provisional_downstream_interface": name if passes else "P0",
                      "note": "Gate is a limited known-wave engineering check, not proof of preservation of all EEG features. If failed, retain P0; candidate results remain reported."})
    save_json("q1_interface_gate.json", gates)

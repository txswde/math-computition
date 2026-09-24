"""Repeat numerical run, audit saved interfaces, and record checks as data."""
from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q1_v7_core import config, folder, load_data, save_json
from q1_v7_deliver import summary
from q1_v7_pipeline import check_protected
from utils import PROJECT_ROOT, sha256_file


def main():
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "pytest", "code/test_q1_v7.py", "-q"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    (folder()/"unit_test_log.txt").write_text(result.stdout+result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Tests failed; see unit_test_log.txt")
    data, meta = load_data()
    audit = json.loads((folder()/"operator_fit_audit.json").read_text(encoding="utf-8"))
    for key, op in audit.items():
        task, partition, method = key.split(":")
        local = meta[meta.task == int(task)]
        validation = local[local.split_role == "test"] if partition == "historical" else \
            local[(local.split_role == "development") & (local.cv_fold == int(partition.split("_")[1]))]
        assert set(validation.trial_key).isdisjoint(op["training_keys"])
        assert set(op["training_keys"]).issubset(set(local[local.split_role == "development"].trial_key))
    with np.load(folder()/"cleaned_epochs.npz") as archive:
        for method in ["P0", "W0", "SELECTED"]:
            x, w = archive[f"{method}_epochs"], archive[f"{method}_weights"]
            assert x.shape == (400, 3, 256) and np.isfinite(x).all()
            assert np.all((w >= .01)&(w <= 1))
            assert np.max(abs(x[..., data["time"] < 0].mean(axis=-1))) < 1e-8
        assert np.max(abs(archive["P0_epochs"]-data["filtered"])) < 1e-12
    with np.load(folder()/"q1_downstream_interface.npz") as archive:
        assert np.array_equal(archive["trial_key"], meta.trial_key.to_numpy())
        assert np.max(abs(archive["signal"]-data["filtered"])) < 1e-12, "Current gate should retain P0"
    trials = pd.read_csv(folder()/"trial_processing.csv")
    assert not trials.duplicated(["trial_key", "method"]).any() and len(trials) == 1200
    assert trials.retained.all()
    primary = pd.read_csv(folder()/"response_statistics.csv")
    primary = primary[primary.primary_family]
    assert len(primary) == 12 and (primary.n_trials_included == 200).all()
    assert (primary.p_holm_conditional >= primary.p_conditional).all()
    summary()
    csv_before = {p.name: sha256_file(p) for p in sorted(folder().glob("*.csv"))}
    repeated = subprocess.run([sys.executable, "code/run_q1_v7.py", "--stage", "all"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True)
    (folder()/"reproduction_log.txt").write_text(repeated.stdout+repeated.stderr, encoding="utf-8")
    if repeated.returncode:
        raise RuntimeError("Numerical repeat failed; see reproduction_log.txt")
    with threadpool_limits(limits=1):
        summary()
    csv_after = {p.name: sha256_file(p) for p in sorted(folder().glob("*.csv"))}
    changed = [name for name in csv_before if csv_after.get(name) != csv_before[name]]
    assert not changed, f"CSV reproduction mismatch: {changed}"
    source = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
              for p in sorted((PROJECT_ROOT/"code").glob("*q1_v7*")) if p.is_file()}
    save_json("verification.json", {"unit_tests": "12 passed", "training_validation_audits": len(audit),
              "trial_method_rows": len(trials), "primary_contrasts": len(primary),
              "numerical_csv_count": len(csv_before), "all_csv_reproduced_exactly": True,
              "csv_sha256": csv_after, "source_sha256": source, "protected": check_protected(),
              "elapsed_s_including_repeat": time.perf_counter()-started,
              "pdf_qa": "Separate visual inspection; not certified by this numerical script",
              "latex": "No XeLaTeX installed; static source check only"})
    print(f"12 tests passed; {len(csv_before)} CSV files reproduced exactly; 79 protected files unchanged.")


if __name__ == "__main__":
    main()

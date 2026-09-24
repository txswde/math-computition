"""Run E0/Q1 stages individually; no Q2/Q3 or final paper is implied."""
import argparse
import time

from threadpoolctl import threadpool_limits

from q1_v7_core import save_json
from q1_v7_pipeline import check_protected, denoise, e0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["e0", "denoise", "statistics", "injection", "curves", "fitting", "sensitivity", "all"], default="e0")
    args = parser.parse_args()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        if args.stage in ["e0", "all"]:
            e0()
        if args.stage in ["denoise", "all"]:
            denoise()
        if args.stage in ["statistics", "all"]:
            from q1_v7_statistics import run_statistics, run_split_half
            run_statistics()
            run_split_half()
        if args.stage in ["injection", "all"]:
            from q1_v7_injection import run
            run()
        if args.stage in ["curves", "all"]:
            from q1_v7_statistics import run_erp_curves
            run_erp_curves()
        if args.stage in ["fitting", "all"]:
            from q1_v7_fitting import run
            run()
        if args.stage in ["sensitivity", "all"]:
            from q1_v7_statistics import run_sensitivity
            run_sensitivity()
        result = check_protected()
        result.update(stage=args.stage, elapsed_s=time.perf_counter()-start)
        save_json(f"run_{args.stage}.json", result)
        print(result, flush=True)


if __name__ == "__main__":
    main()

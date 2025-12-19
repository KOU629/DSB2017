import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple


METRIC_KEYS = [
    "cases_evaluated",
    "gt_total",
    "tp_total",
    "fp_total",
    "fn_total",
    "sensitivity",
    "fp_per_scan",
]


def run_eval(ids_file: str, conf_th: float, detect_th: float, nms_th: float,
             only_positive: bool = True) -> Dict[str, float]:
    cmd = [
        sys.executable,
        os.path.join("tools", "eval_pbb.py"),
        "--ids-file", ids_file,
        "--conf-th", str(conf_th),
        "--detect-th", str(detect_th),
        "--nms-th", str(nms_th),
    ]
    if only_positive:
        cmd.append("--only-positive-labels")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    out = proc.stdout + "\n" + proc.stderr

    metrics: Dict[str, float] = {}
    # Parse the lines printed by eval_pbb.py
    # Expected lines like: key=value or key= number
    patterns = {
        "cases_evaluated": re.compile(r"cases_evaluated=\s*(\d+)")
    }
    for k in ["gt_total", "tp_total", "fp_total", "fn_total"]:
        patterns[k] = re.compile(rf"{k}=\s*(\d+)")
    patterns["sensitivity"] = re.compile(r"sensitivity=\s*([0-9.]+)")
    patterns["fp_per_scan"] = re.compile(r"fp_per_scan=\s*([0-9.]+)")

    for key, pat in patterns.items():
        m = pat.search(out)
        if m:
            val = float(m.group(1))
            # ints for count metrics
            if key in {"cases_evaluated", "gt_total", "tp_total", "fp_total", "fn_total"}:
                val = int(val)
            metrics[key] = val

    if "cases_evaluated" not in metrics:
        raise RuntimeError(f"Failed to parse eval output for conf_th={conf_th}, detect_th={detect_th}:\n{out}")

    return metrics


def choose_best(results: List[Tuple[float, float, Dict[str, float]]]) -> Tuple[float, float, Dict[str, float]]:
    # Policy: prefer lowest fp_per_scan among those with sensitivity >= 0.90; if none, pick highest sensitivity
    eligible = [r for r in results if r[2].get("sensitivity", 0.0) >= 0.90]
    if eligible:
        eligible.sort(key=lambda x: (x[2].get("fp_per_scan", 1e9), -x[2].get("sensitivity", 0.0)))
        return eligible[0]
    # fallback: highest sensitivity, tie-breaker: lower fp_per_scan
    results.sort(key=lambda x: (-x[2].get("sensitivity", 0.0), x[2].get("fp_per_scan", 1e9)))
    return results[0]


def main():
    parser = argparse.ArgumentParser(description="Sweep eval_pbb thresholds and summarize metrics")
    parser.add_argument("--ids-file", default="tools/ids_valid.txt")
    parser.add_argument("--conf-th", nargs="*", type=float, default=[-2.0, -1.8, -1.5, -1.3, -1.2, -1.0, -0.8])
    parser.add_argument("--detect-th", nargs="*", type=float, default=[0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--nms-th", type=float, default=0.1)
    parser.add_argument("--only-positive-labels", action="store_true")
    parser.add_argument("--out-csv", default="tools/sweep_eval_results.csv")
    args = parser.parse_args()

    grid = list(itertools.product(sorted(set(args.conf_th)), sorted(set(args.detect_th))))
    print(f"sweep size: {len(grid)}")

    results: List[Tuple[float, float, Dict[str, float]]] = []
    for i, (c, d) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] conf_th={c}, detect_th={d}")
        metrics = run_eval(args.ids_file, c, d, args.nms_th, only_positive=args.only_positive_labels)
        results.append((c, d, metrics))

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["conf_th", "detect_th"] + METRIC_KEYS
        writer.writerow(header)
        for c, d, m in results:
            row = [c, d] + [m.get(k, "") for k in METRIC_KEYS]
            writer.writerow(row)

    best_c, best_d, best_m = choose_best(results)
    print("\n=== Best Recommendation ===")
    print(f"conf_th={best_c}, detect_th={best_d}")
    for k in METRIC_KEYS:
        if k in best_m:
            print(f"{k}={best_m[k]}")
    print(f"CSV written: {args.out_csv}")


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List


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
             only_positive: bool, bbox_dir: str) -> Dict[str, float]:
    cmd = [
        sys.executable,
        os.path.join("tools", "eval_pbb.py"),
        "--ids-file", ids_file,
        "--bbox-dir", bbox_dir,
        "--conf-th", str(conf_th),
        "--detect-th", str(detect_th),
        "--nms-th", str(nms_th),
    ]
    if only_positive:
        cmd.append("--only-positive-labels")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    out = proc.stdout + "\n" + proc.stderr

    patterns = {
        "cases_evaluated": re.compile(r"cases_evaluated=\s*(\d+)"),
        "gt_total": re.compile(r"gt_total=\s*(\d+)"),
        "tp_total": re.compile(r"tp_total=\s*(\d+)"),
        "fp_total": re.compile(r"fp_total=\s*(\d+)"),
        "fn_total": re.compile(r"fn_total=\s*(\d+)"),
        "sensitivity": re.compile(r"sensitivity=\s*([0-9.]+)"),
        "fp_per_scan": re.compile(r"fp_per_scan=\s*([0-9.]+)"),
    }

    metrics: Dict[str, float] = {}
    for key, pat in patterns.items():
        m = pat.search(out)
        if not m:
            continue
        val: float
        if key in {"cases_evaluated", "gt_total", "tp_total", "fp_total", "fn_total"}:
            val = int(m.group(1))
        else:
            val = float(m.group(1))
        metrics[key] = val

    if "cases_evaluated" not in metrics:
        raise RuntimeError(f"Failed to parse eval output for conf_th={conf_th}:\n{out}")

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Generate FROC input by sweeping conf_th with fixed detect_th (matching criterion)."
    )
    ap.add_argument("--ids-file", default="tools/all_ids.txt")
    ap.add_argument("--bbox-dir", default="bbox_result")
    ap.add_argument("--conf-th", nargs="*", type=float,
                    default=[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--detect-th", type=float, default=0.1,
                    help="Fixed IoU threshold for TP matching (evaluation protocol; do NOT sweep for FROC).")
    ap.add_argument("--nms-th", type=float, default=0.1)
    ap.add_argument("--only-positive-labels", action="store_true")
    ap.add_argument("--out-csv", default="tools/sweep_froc_results.csv")
    args = ap.parse_args()

    conf_list = sorted(set(args.conf_th))
    results: List[Dict[str, float]] = []

    print(f"sweep size: {len(conf_list)} (detect_th fixed at {args.detect_th})")
    for i, c in enumerate(conf_list, 1):
        print(f"[{i}/{len(conf_list)}] conf_th={c}")
        m = run_eval(
            ids_file=args.ids_file,
            conf_th=c,
            detect_th=args.detect_th,
            nms_th=args.nms_th,
            only_positive=bool(args.only_positive_labels),
            bbox_dir=args.bbox_dir,
        )
        m["conf_th"] = c
        m["detect_th"] = args.detect_th
        results.append(m)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conf_th", "detect_th"] + METRIC_KEYS)
        for m in results:
            writer.writerow([m["conf_th"], m["detect_th"]] + [m.get(k, "") for k in METRIC_KEYS])

    print(f"CSV written: {args.out_csv}")


if __name__ == "__main__":
    main()

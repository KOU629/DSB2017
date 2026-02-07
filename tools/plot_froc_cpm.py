import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D


CPM_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def _nice_float(x: float) -> str:
    # Avoid labels like 0.1000000003
    if abs(x) < 1e-12:
        return "0"
    if x >= 1:
        return f"{x:g}"
    return f"{x:.3g}"


def load_sweep_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("fp_per_scan") and r.get("sensitivity"):
                rows.append(r)
    return rows


def points_from_rows(rows):
    points = []
    for r in rows:
        points.append((float(r["fp_per_scan"]), float(r["sensitivity"])))
    return points


def froc_curve(points, fp_round: int = 4):
    # Aggregate by FP/scan (rounded): keep the best sensitivity per FP/scan.
    # Rounding avoids float-key instability across runs/platforms.
    best = {}
    for fp, sen in points:
        fp_key = round(float(fp), fp_round)
        if fp_key not in best or sen > best[fp_key]:
            best[fp_key] = float(sen)

    pts = sorted(best.items(), key=lambda x: x[0])
    xs = np.array([p[0] for p in pts], dtype=float)
    ys_best = np.array([p[1] for p in pts], dtype=float)
    # Monotonic envelope (standard FROC): best sensitivity achievable at or below FP
    ys_env = np.maximum.accumulate(ys_best)
    return xs, ys_best, ys_env


def cpm(xs, ys, cpm_points=CPM_POINTS, require_coverage: bool = True):
    # Conservative rule: if FROC doesn't reach a low-FP region, sensitivity there is 0.
    # For thesis reporting, it's safer to require coverage up to max CPM FP/scan.
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return 0.0
    max_need = float(max(cpm_points))
    if require_coverage and float(xs.max()) < max_need:
        raise ValueError(
            f"FROC does not cover CPM max FP/scan={max_need} (max FP/scan in sweep={float(xs.max()):.4f}). "
            f"Extend conf_th sweep to reach >= {max_need}."
        )
    return float(np.mean(np.interp(cpm_points, xs, ys, left=0.0, right=ys[-1])))


def find_op_point(rows, conf_th, detect_th):
    for r in rows:
        if math.isclose(float(r["conf_th"]), float(conf_th), rel_tol=0.0, abs_tol=1e-9) and math.isclose(
            float(r["detect_th"]), float(detect_th), rel_tol=0.0, abs_tol=1e-9
        ):
            return r
    return None


def choose_recommended_op(rows, min_sens=0.90):
    parsed = []
    for r in rows:
        try:
            parsed.append({
                "conf_th": float(r["conf_th"]),
                "detect_th": float(r["detect_th"]),
                "sensitivity": float(r["sensitivity"]),
                "fp_per_scan": float(r["fp_per_scan"]),
                "row": r,
            })
        except Exception:
            continue
    if not parsed:
        return None

    eligible = [p for p in parsed if p["sensitivity"] >= min_sens]
    if eligible:
        eligible.sort(key=lambda p: (p["fp_per_scan"], -p["sensitivity"]))
        best = eligible[0]
    else:
        parsed.sort(key=lambda p: (-p["sensitivity"], p["fp_per_scan"]))
        best = parsed[0]

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", nargs="+", default=["tools/sweep_eval_results.csv"])
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--op-point", nargs="*", default=None,
                    help="Pairs of conf_th detect_th for each input CSV, e.g., --op-point -0.8 0.35 -1.2 0.4")
    ap.add_argument("--out-png", default="tools/froc.png")
    ap.add_argument("--out-csv", default="tools/froc_points.csv")
    ap.add_argument("--out-json", default="tools/froc_summary.json")
    ap.add_argument("--out-table-csv", default="tools/froc_table.csv")
    ap.add_argument("--out-table-md", default="tools/froc_table.md")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--nms-th", type=float, default=None)
    ap.add_argument("--only-positive-labels", action="store_true")
    ap.add_argument("--sweep-conf", nargs="*", type=float, default=None)
    ap.add_argument("--sweep-detect", nargs="*", type=float, default=None)
    ap.add_argument("--min-sens", type=float, default=0.90, help="Minimum sensitivity for recommended operating point")
    ap.add_argument("--fp-round", type=int, default=4, help="Decimal rounding for FP/scan grouping in envelope")
    ap.add_argument(
        "--require-cpm-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require sweep to cover up to max CPM FP/scan (default: true)",
    )
    ap.add_argument("--xscale", choices=["log", "linear"], default="log")
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    ap.add_argument(
        "--xticks",
        nargs="*",
        type=float,
        default=None,
        help="Explicit x ticks. Example: --xticks 0.1 0.2 0.5 1 2 5 10",
    )
    ap.add_argument(
        "--xticks-include-cpm",
        action="store_true",
        help="Include CPM points in x ticks (useful on log axis).",
    )
    ap.add_argument(
        "--show-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show raw sweep points on the plot (default: true)",
    )
    ap.add_argument(
        "--show-cpm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show CPM interpolation points on the plot (default: true)",
    )
    ap.add_argument(
        "--cpm-color",
        type=str,
        default="tab:red",
        help="Color for CPM points (default: tab:red)",
    )
    ap.add_argument(
        "--legend-style",
        choices=["auto", "kind", "run"],
        default="auto",
        help="Legend style. auto: kind-only for 1 run, run-only for multi-run.",
    )
    ap.add_argument(
        "--x-tick-rotation",
        type=float,
        default=0.0,
        help="Rotate x tick labels (degrees) to avoid overlap (default: 0).",
    )
    ap.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Optional caption text to draw on the plot (e.g., 'NMS=0.1, IoU≥0.35').",
    )
    ap.add_argument(
        "--caption-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate a plot caption for single-run plots (default: true).",
    )
    args = ap.parse_args()

    labels = args.label or []
    if labels and len(labels) != len(args.in_csv):
        raise ValueError("--label count must match --in-csv count")
    if not labels:
        labels = [f"run{i+1}" for i in range(len(args.in_csv))]

    op_points = None
    if args.op_point:
        if len(args.op_point) % 2 != 0:
            raise ValueError("--op-point must be pairs of conf_th detect_th")
        pairs = list(zip(args.op_point[0::2], args.op_point[1::2]))
        if len(pairs) != len(args.in_csv):
            raise ValueError("--op-point pairs must match --in-csv count")
        op_points = [(float(c), float(d)) for c, d in pairs]

    summary = {
        "ids_file": args.ids_file,
        "nms_th": args.nms_th,
        "only_positive_labels": bool(args.only_positive_labels),
        "sweep_conf": args.sweep_conf,
        "sweep_detect": args.sweep_detect,
        "cpm_points": CPM_POINTS,
        "fp_round": args.fp_round,
        "min_sens": args.min_sens,
        "require_cpm_coverage": bool(args.require_cpm_coverage),
        "runs": [],
    }

    plt.figure(figsize=(5.5, 4.5))
    all_env_points = []
    all_raw_points = []
    all_raw_rows = []

    n_runs = len(args.in_csv)
    if args.legend_style == "auto":
        legend_style = "kind" if n_runs == 1 else "run"
    else:
        legend_style = args.legend_style

    plot_caption = args.caption
    if plot_caption is None and args.caption_auto and n_runs == 1:
        parts = []
        if args.nms_th is not None:
            parts.append(f"NMS={args.nms_th}")
        # IoU threshold is detect_th in this repo's evaluation.
        # Prefer reading it from the sweep rows if consistent; else fall back to op point.
        try:
            detect_vals = sorted({str(r.get("detect_th")) for r in load_sweep_csv(args.in_csv[0]) if r.get("detect_th")})
            detect_vals = [v for v in detect_vals if v not in ("", "None")]
        except Exception:
            detect_vals = []
        if len(detect_vals) == 1:
            parts.append(f"IoU≥{detect_vals[0]}")
        plot_caption = ", ".join(parts) if parts else None

    envelope_handles_for_run_legend = []
    envelope_labels_for_run_legend = []

    for i, (csv_path, label) in enumerate(zip(args.in_csv, labels)):
        rows = load_sweep_csv(csv_path)
        points = points_from_rows(rows)
        if not points:
            raise RuntimeError(f"No points found in {csv_path}")

        # Keep true raw sweep rows for export (matches the scatter).
        for r in rows:
            all_raw_rows.append(
                (
                    label,
                    float(r.get("conf_th", 0.0)) if r.get("conf_th") not in (None, "") else "",
                    float(r.get("detect_th", 0.0)) if r.get("detect_th") not in (None, "") else "",
                    float(r["fp_per_scan"]),
                    float(r["sensitivity"]),
                )
            )

        raw_xs = np.array([p[0] for p in points], dtype=float)
        raw_ys = np.array([p[1] for p in points], dtype=float)

        xs, ys_best, ys_env = froc_curve(points, fp_round=args.fp_round)
        cpm_value = cpm(xs, ys_env, require_coverage=bool(args.require_cpm_coverage))
        cpm_y = np.interp(CPM_POINTS, xs, ys_env, left=0.0, right=ys_env[-1])

        # Guard: log-scale can't display FP/scan <= 0. Filter only for plotting (metrics use unfiltered data).
        raw_mask = raw_xs > 0
        if not bool(np.all(raw_mask)):
            print(f"WARN: {label} has {int((~raw_mask).sum())} raw points with FP/scan<=0; excluded from plot.")
        env_mask = xs > 0
        if not bool(np.all(env_mask)):
            print(f"WARN: {label} has {int((~env_mask).sum())} envelope points with FP/scan<=0; excluded from plot.")

        # Legend labels
        if legend_style == "kind":
            raw_label = "Sweep points"
            env_label = "FROC envelope"
            cpm_label = "CPM points"
        else:
            raw_label = None
            env_label = label
            cpm_label = None

        if args.show_raw and bool(np.any(raw_mask)):
            plt.scatter(
                raw_xs[raw_mask],
                raw_ys[raw_mask],
                s=14,
                alpha=0.25,
                label=raw_label if raw_label else "_nolegend_",
            )

        if bool(np.any(env_mask)):
            line_handle, = plt.plot(
                xs[env_mask],
                ys_env[env_mask],
                marker="o",
                linewidth=1.6,
                label=env_label if env_label else "_nolegend_",
            )
            if legend_style == "run":
                envelope_handles_for_run_legend.append(line_handle)
                envelope_labels_for_run_legend.append(label)
        else:
            print(f"WARN: {label} has no envelope points with FP/scan>0; skipping envelope plot.")

        if args.show_cpm:
            plt.scatter(
                CPM_POINTS,
                cpm_y,
                s=28,
                zorder=3,
                color=args.cpm_color,
                label=cpm_label if cpm_label else "_nolegend_",
            )

        op_row = None
        op_point = None
        op_policy = None
        if op_points:
            op_point = op_points[i]
            op_row = find_op_point(rows, *op_point)
            op_policy = "fixed"
        else:
            best = choose_recommended_op(rows, min_sens=args.min_sens)
            if best is not None:
                op_point = (best["conf_th"], best["detect_th"])
                op_row = best["row"]
                op_policy = f"min_fp_at_sens>={args.min_sens:.2f}" if best["sensitivity"] >= args.min_sens else "max_sensitivity"

        summary["runs"].append({
            "label": label,
            "in_csv": csv_path,
            "cpm": cpm_value,
            "operating_point": op_point,
            "operating_point_policy": op_policy,
            "operating_metrics": op_row,
        })

        for x, y_best, y_env in zip(xs, ys_best, ys_env):
            all_env_points.append((label, x, y_env))
            all_raw_points.append((label, x, y_best))
    # === Axis configuration ===
    if args.xscale == "log":
        plt.xscale("log")

        if args.xticks is not None and len(args.xticks) > 0:
            ticks = args.xticks[:]
        else:
            # Default ticks: keep modest count to avoid label overlap.
            ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
            if args.xticks_include_cpm:
                # Prefer CPM points only (avoid crowded 0.1 vs 0.125 labels).
                ticks = list(CPM_POINTS)

        # Log axis cannot show non-positive ticks.
        ticks = [t for t in ticks if t > 0]
        if ticks:
            plt.xticks(ticks)
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _nice_float(float(x))))
            plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            plt.gca().xaxis.set_minor_formatter(NullFormatter())
    else:
        plt.xscale("linear")
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _nice_float(float(x))))

        # If the user wants CPM ticks in linear mode and didn't provide explicit ticks,
        # prefer CPM points (+ optional 0/10 for context) to avoid clutter.
        if (args.xticks is None or len(args.xticks) == 0) and args.xticks_include_cpm:
            ticks = list(CPM_POINTS)
            if args.xlim:
                xmin, xmax = args.xlim
                if xmin <= 0 <= xmax:
                    ticks = [0.0] + ticks
                if xmin <= 10 <= xmax:
                    ticks = ticks + [10.0]
            plt.xticks(sorted(set(ticks)))

    if args.xlim:
        plt.xlim(args.xlim[0], args.xlim[1])

    if args.x_tick_rotation:
        plt.gca().tick_params(axis="x", labelrotation=args.x_tick_rotation)
    plt.xlabel("FP/scan")
    plt.ylabel("Sensitivity")
    plt.title("FROC")
    if plot_caption:
        # Put caption inside the figure area so it survives copy/paste into thesis.
        plt.gcf().text(0.01, 0.01, plot_caption, ha="left", va="bottom", fontsize=8)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    # Legend rendering
    if legend_style == "kind":
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(loc="lower right", frameon=False, fontsize=8)
    else:
        # Two legends: runs (envelope lines) + type legend (proxies)
        if envelope_handles_for_run_legend:
            leg1 = plt.legend(
                envelope_handles_for_run_legend,
                envelope_labels_for_run_legend,
                loc="lower right",
                frameon=False,
                fontsize=8,
                title="Run",
            )
            plt.gca().add_artist(leg1)

        type_handles = []
        type_labels = []
        if args.show_raw:
            type_handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=4, alpha=0.5))
            type_labels.append("Sweep points")
        type_handles.append(Line2D([0], [0], marker="o", linestyle="-", linewidth=1.6, markersize=4))
        type_labels.append("FROC envelope")
        if args.show_cpm:
            type_handles.append(
                Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor=args.cpm_color, markeredgecolor=args.cpm_color)
            )
            type_labels.append("CPM points")
        plt.legend(type_handles, type_labels, loc="lower left", frameon=False, fontsize=8, title="Type")
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "type", "conf_th", "detect_th", "fp_per_scan", "sensitivity"])
        for label, conf_th, detect_th, fp, sen in all_raw_rows:
            writer.writerow([label, "raw", conf_th, detect_th, fp, sen])
        for label, x, y_env in all_env_points:
            writer.writerow([label, "envelope", "", "", x, y_env])
        for label, x, y_best in all_raw_points:
            writer.writerow([label, "best_at_fp", "", "", x, y_best])

    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write thesis-ready tables
    table_header = [
        "label",
        "cpm",
        "conf_th",
        "detect_th",
        "sensitivity",
        "fp_per_scan",
        "tp_total",
        "fn_total",
        "fp_total",
        "gt_total",
        "cases_evaluated",
        "op_policy",
    ]

    table_rows = []
    for run in summary["runs"]:
        op = run.get("operating_point")
        metrics = run.get("operating_metrics") or {}
        conf_th = op[0] if op else ""
        detect_th = op[1] if op else ""
        table_rows.append([
            run.get("label", ""),
            float(run.get("cpm", 0.0)),
            conf_th,
            detect_th,
            float(metrics.get("sensitivity", 0.0)) if metrics.get("sensitivity") not in (None, "") else "",
            float(metrics.get("fp_per_scan", 0.0)) if metrics.get("fp_per_scan") not in (None, "") else "",
            metrics.get("tp_total", ""),
            metrics.get("fn_total", ""),
            metrics.get("fp_total", ""),
            metrics.get("gt_total", ""),
            metrics.get("cases_evaluated", ""),
            run.get("operating_point_policy", ""),
        ])

    Path(args.out_table_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(table_header)
        for r in table_rows:
            writer.writerow(r)

    Path(args.out_table_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_table_md, "w", encoding="utf-8") as f:
        f.write("# FROC/CPM Summary\n\n")
        # Make the image path relative to the markdown file so it renders in VS Code preview.
        md_dir = Path(args.out_table_md).resolve().parent
        png_path = Path(args.out_png).resolve()
        rel_png = os.path.relpath(str(png_path), str(md_dir)).replace("\\", "/")
        f.write(f"![FROC plot]({rel_png})\n\n")
        ids_file = summary.get("ids_file")
        ids_display = Path(ids_file).name if ids_file else None
        f.write(f"- ids_file: {ids_display or ids_file}\n")
        f.write(f"- nms_th: {summary.get('nms_th')}\n")
        # Helpful for thesis captions: show IoU threshold if it's consistent.
        detect_th_vals = sorted({str(r[3]) for r in table_rows if str(r[3]) not in ("", "None")})
        if len(detect_th_vals) == 1:
            f.write(f"- IoU≥ (detect_th): {detect_th_vals[0]}\n")
        elif len(detect_th_vals) > 1:
            f.write(f"- IoU≥ (detect_th): {detect_th_vals}\n")
        f.write(f"- only_positive_labels: {summary.get('only_positive_labels')}\n")
        f.write(f"- cpm_points: {summary.get('cpm_points')}\n\n")
        f.write("| " + " | ".join(table_header) + " |\n")
        f.write("|" + "|".join(["---"] * len(table_header)) + "|\n")
        for r in table_rows:
            f.write("| " + " | ".join([str(x) for x in r]) + " |\n")

    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_table_csv}")
    print(f"Saved: {args.out_table_md}")

    for run in summary["runs"]:
        op = run.get("operating_point")
        metrics = run.get("operating_metrics") or {}
        if op:
            print(
                f"{run['label']}: CPM={run['cpm']:.4f}, op(conf_th={op[0]}, detect_th={op[1]}), "
                f"sens={metrics.get('sensitivity')}, fp/scan={metrics.get('fp_per_scan')}"
            )
        else:
            print(f"{run['label']}: CPM={run['cpm']:.4f}")


if __name__ == "__main__":
    main()

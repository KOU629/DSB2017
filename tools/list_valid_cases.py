import argparse
import glob
import os
import sys
from typing import List

import numpy as np


def find_valid_cases(bbox_dir: str, min_diameter: float = 0.0) -> List[str]:
    pattern = os.path.join(bbox_dir, "*_lbb.npy")
    ids: List[str] = []
    for lbb_path in sorted(glob.glob(pattern)):
        try:
            arr = np.load(lbb_path)
        except Exception as e:
            print(f"[WARN] Failed to load {lbb_path}: {e}", file=sys.stderr)
            continue
        if arr.size == 0:
            continue
        # Expect shape (N, 4) with last column being diameter
        try:
            has_positive = np.any(arr[:, 3] > min_diameter)
        except Exception:
            # If unexpected shape, skip
            continue
        if has_positive:
            case_id = os.path.basename(lbb_path).replace("_lbb.npy", "")
            ids.append(case_id)
    return ids


def main():
    parser = argparse.ArgumentParser(description="List cases with positive ground-truth diameter and write to file")
    parser.add_argument("--bbox-dir", default="bbox_result", help="Directory containing *_lbb.npy and *_pbb.npy")
    parser.add_argument("--out", default="tools/ids_valid.txt", help="Output file to write the case IDs")
    parser.add_argument("--min-diameter", type=float, default=0.0, help="Minimum diameter to consider a label valid")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ids = find_valid_cases(args.bbox_dir, args.min_diameter)

    # Write out, one per line
    with open(args.out, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(f"{cid}\n")

    print(f"valid cases with positive labels: {len(ids)}")
    if ids:
        print("first 20:", ids[:20])
    print(f"written to: {args.out}")


if __name__ == "__main__":
    main()

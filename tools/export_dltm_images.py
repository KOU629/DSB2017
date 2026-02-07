import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export one full CT slice and 5 cropped candidate images for DLTM figure."
    )
    parser.add_argument("--clean-npy", required=True, help="Path to *_clean.npy")
    parser.add_argument("--pbb-npy", required=True, help="Path to *_pbb.npy")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--topk", type=int, default=5, help="Number of candidates to export")
    parser.add_argument("--conf-th", type=float, default=-1.0, help="Score threshold for pbb")
    parser.add_argument("--nms-th", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--slice-z", type=int, default=None, help="Optional fixed z-slice for full image")
    parser.add_argument("--slice-tol", type=int, default=1, help="Z tolerance for drawing boxes")
    parser.add_argument("--crop-scale", type=float, default=1.0, help="Scale for auto crop size from median r")
    parser.add_argument("--crop-size", type=int, default=256, help="Resize crop to NxN pixels")
    parser.add_argument("--crop-half", type=int, default=0, help="Fixed half-size in voxels (0=auto)")
    return parser.parse_args()


def normalize_ct(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() > 255 or img.min() < 0:
        img = np.clip(img, -1200, 600)
        img = (img + 1200) / 1800.0 * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def iou_cube(a, b):
    # a,b: [z,y,x,r]
    az0, ay0, ax0, ar = a
    bz0, by0, bx0, br = b
    a_min = np.array([az0 - ar, ay0 - ar, ax0 - ar])
    a_max = np.array([az0 + ar, ay0 + ar, ax0 + ar])
    b_min = np.array([bz0 - br, by0 - br, bx0 - br])
    b_max = np.array([bz0 + br, by0 + br, bx0 + br])

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter[0] * inter[1] * inter[2]
    a_vol = (2 * ar) ** 3
    b_vol = (2 * br) ** 3
    union = a_vol + b_vol - inter_vol
    if union <= 0:
        return 0.0
    return inter_vol / union


def nms_pbb(pbb, iou_th):
    if len(pbb) == 0:
        return pbb
    order = np.argsort(-pbb[:, 0])
    keep = []
    suppressed = np.zeros(len(pbb), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or j == i:
                continue
            if iou_cube(pbb[i, 1:5], pbb[j, 1:5]) > iou_th:
                suppressed[j] = True
    return pbb[keep]


def save_full_slice(ct, candidates, out_path, slice_z=None, slice_tol=1, half_size=None):
    if slice_z is None:
        slice_z = int(round(candidates[0, 1]))
    slice_z = int(np.clip(slice_z, 0, ct.shape[0] - 1))

    if half_size is None:
        half_size = int(round(np.median(candidates[:, 4])))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(ct[slice_z], cmap="gray", interpolation="bilinear")
    ax.axis("off")

    for c in candidates:
        z, y, x, r = c[1:5]
        if abs(z - slice_z) <= slice_tol:
            r = int(round(half_size))
            x = int(round(x))
            y = int(round(y))
            rect = plt.Rectangle(
                (x - r, y - r),
                2 * r,
                2 * r,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_crops(ct, candidates, out_dir, crop_scale=1.0, crop_size=256, crop_half=0):
    out_dir = Path(out_dir)
    if crop_half and crop_half > 0:
        half_size = int(crop_half)
    else:
        half_size = int(round(np.median(candidates[:, 4]) * crop_scale))
    for i, c in enumerate(candidates, start=1):
        z, y, x, r = c[1:5]
        z = int(round(z))
        y = int(round(y))
        x = int(round(x))
        r = int(half_size)

        z = int(np.clip(z, 0, ct.shape[0] - 1))
        y0 = max(0, y - r)
        y1 = min(ct.shape[1], y + r)
        x0 = max(0, x - r)
        x1 = min(ct.shape[2], x + r)

        crop = ct[z, y0:y1, x0:x1]
        if crop.size == 0:
            continue
        img = Image.fromarray(crop)
        if crop_size is not None and crop_size > 0:
            img = img.resize((crop_size, crop_size), resample=Image.Resampling.LANCZOS)
        img.save(out_dir / f"candidate_{i:02d}.png")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = np.load(args.clean_npy)
    if ct.ndim == 4:
        ct = ct[0]
    ct = normalize_ct(ct)

    pbb = np.load(args.pbb_npy)
    if pbb.ndim != 2 or pbb.shape[1] < 5:
        raise ValueError("pbb must be shape [N,5+] with columns [score,z,y,x,r]")
    pbb = pbb[pbb[:, 0] > args.conf_th]
    if len(pbb) == 0:
        raise ValueError("No candidates after conf threshold.")

    pbb = nms_pbb(pbb, args.nms_th)
    pbb = pbb[np.argsort(-pbb[:, 0])]
    if len(pbb) > args.topk:
        pbb = pbb[: args.topk]

    save_full_slice(
        ct,
        pbb,
        out_dir / "full_slice.png",
        slice_z=args.slice_z,
        slice_tol=args.slice_tol,
        half_size=args.crop_half if args.crop_half > 0 else None,
    )
    save_crops(
        ct,
        pbb,
        out_dir,
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        crop_half=args.crop_half,
    )

    print(f"Saved: {out_dir / 'full_slice.png'}")
    print(f"Saved: {out_dir / 'candidate_01.png'} ...")


if __name__ == "__main__":
    main()

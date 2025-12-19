import os
import argparse
import numpy as np
import sys

# Import iou and nms from layers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from layers import iou, nms


def load_ids(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]


def filter_valid_labels(lbb: np.ndarray) -> np.ndarray:
    if lbb is None or lbb.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lbb = np.asarray(lbb)
    if lbb.ndim == 1 and lbb.size == 4:
        lbb = lbb.reshape(1, 4)
    # keep diameter > 0
    lbb = lbb[lbb[:, 3] > 0]
    if lbb.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return lbb.astype(np.float32)


def label_proposals(pbb: np.ndarray, lbb: np.ndarray, detect_th: float) -> np.ndarray:
    y = np.zeros((len(pbb),), dtype=np.int32)
    if len(pbb) == 0 or len(lbb) == 0:
        return y
    for i, p in enumerate(pbb):
        best = 0.0
        for l in lbb:
            best = max(best, iou(p[1:5], l))
            if best > detect_th:
                break
        y[i] = 1 if best > detect_th else 0
    return y


def lda_probe_auc(X: np.ndarray, y: np.ndarray) -> float:
    # y in {0,1}
    y = y.astype(np.int32)
    pos = X[y == 1]
    neg = X[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    m1 = pos.mean(axis=0)
    m0 = neg.mean(axis=0)

    def _cov(a):
        if len(a) < 2:
            return np.zeros((a.shape[1], a.shape[1]), dtype=np.float64)
        return np.cov(a.T)

    Sw = _cov(pos) + _cov(neg) + 1e-3 * np.eye(X.shape[1])
    w = np.linalg.pinv(Sw) @ (m1 - m0)
    scores = (X @ w).ravel()
    # AUC via rank statistic (Mann–Whitney U)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    R1 = ranks[y == 1].sum()
    auc = (R1 - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def main():
    ap = argparse.ArgumentParser(description='Evaluate proposal purity and optional feature probe AUC')
    ap.add_argument('--ids-file', required=True, help='Path to IDs file (e.g., tools/ids_valid.txt)')
    ap.add_argument('--bbox-dir', default='bbox_result', help='Dir with *_pbb.npy, *_lbb.npy, *_feature.npy')
    ap.add_argument('--nms-th', type=float, default=-1.0, help='NMS IoU threshold for proposals (<=0 to disable)')
    ap.add_argument('--detect-th', type=float, default=0.35, help='IoU threshold for positive label')
    ap.add_argument('--probe', action='store_true', help='Compute LDA-based linear probe AUC if features available')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of cases')
    args = ap.parse_args()

    ids = load_ids(args.ids_file)
    if args.limit and args.limit > 0:
        ids = ids[:args.limit]

    total_props = 0
    total_pos = 0
    per_case_pos = []

    X_list = []
    y_list = []
    used_cases_probe = 0

    for cid in ids:
        pbb_path = os.path.join(args.bbox_dir, f'{cid}_pbb.npy')
        lbb_path = os.path.join(args.bbox_dir, f'{cid}_lbb.npy')
        if not (os.path.exists(pbb_path) and os.path.exists(lbb_path)):
            continue
        try:
            pbb = np.load(pbb_path)
            lbb = filter_valid_labels(np.load(lbb_path))
        except Exception:
            continue

        # Optionally NMS (note: applying NMS may break length match with features)
        if pbb.size == 0:
            pbb = np.zeros((0, 5), dtype=np.float32)
        elif args.nms_th and args.nms_th > 0:
            pbb = nms(pbb, args.nms_th)

        y = label_proposals(pbb, lbb, args.detect_th)
        total_props += len(pbb)
        total_pos += int(y.sum())
        per_case_pos.append(int(y.sum()))

        if args.probe:
            feat_path = os.path.join(args.bbox_dir, f'{cid}_feature.npy')
            if os.path.exists(feat_path):
                try:
                    X = np.load(feat_path)
                    # Expect len(X) == len(pbb). If not, try to align by skipping.
                    if len(X) == len(pbb) and len(X) > 0:
                        if X.ndim > 2:
                            X = X.reshape((X.shape[0], -1))
                        X_list.append(X.astype(np.float64))
                        y_list.append(y.astype(np.int32))
                        used_cases_probe += 1
                except Exception:
                    pass

    cases = len(per_case_pos)
    pos_ratio = (total_pos / total_props) if total_props else 0.0
    per_case_mean = (np.mean(per_case_pos) if per_case_pos else 0.0)
    per_case_median = (np.median(per_case_pos) if per_case_pos else 0.0)

    print(f'cases_evaluated={cases}')
    print(f'total_proposals={total_props}')
    print(f'positive_proposals={total_pos}')
    print(f'pos_ratio={pos_ratio:.4f}')
    print(f'per_case_pos_mean={per_case_mean:.3f}')
    print(f'per_case_pos_median={per_case_median:.3f}')

    if args.probe and X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        # standardize
        X_all = X_all - X_all.mean(axis=0, keepdims=True)
        auc = lda_probe_auc(X_all, y_all)
        print(f'probe_cases_used={used_cases_probe}')
        print(f'probe_samples={len(y_all)} (pos={int(y_all.sum())}, neg={int(len(y_all)-y_all.sum())})')
        print(f'probe_auc={auc:.4f}')


if __name__ == '__main__':
    main()

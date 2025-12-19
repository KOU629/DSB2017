import os
import glob
import argparse
import numpy as np

def load_ids_from_file(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description='Evaluate PBB vs LBB across cases')
    parser.add_argument('--ids-file', type=str, default='', help='Path to a text file with one case ID per line')
    parser.add_argument('--conf-th', type=float, default=-2.0, help='Confidence threshold used in evaluation')
    parser.add_argument('--nms-th', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--detect-th', type=float, default=0.35, help='Detection IoU threshold to count TP')
    parser.add_argument('--only-positive-labels', action='store_true', help='Skip cases with no positive-diameter labels')
    args = parser.parse_args()

    # Import acc from layers
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from layers import acc

    p_paths = []
    if args.ids_file:
        ids = load_ids_from_file(args.ids_file)
        for cid in ids:
            p = os.path.join(os.path.dirname(__file__), '..', 'bbox_result', f'{cid}_pbb.npy')
            if os.path.exists(p):
                p_paths.append(p)
    else:
        p_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'bbox_result', '*_pbb.npy')))

    stats = {'tp':0,'fp':0,'fn':0,'gt':0,'scans':0}
    used = []
    for p in p_paths:
        base = os.path.basename(p).replace('_pbb.npy','')
        l = os.path.join(os.path.dirname(__file__), '..', 'bbox_result', f'{base}_lbb.npy')
        if not os.path.exists(l):
            continue
        lbb = np.load(l)
        lbb = np.asarray(lbb, dtype=np.float32)
        if lbb.ndim == 1 and lbb.shape[0] == 4:
            lbb = lbb.reshape(1, 4)
        # Drop invalid labels globally: keep only diameter > 0
        if lbb.size:
            if lbb.shape[1] >= 4:
                lbb = lbb[lbb[:, 3] > 0]
            else:
                lbb = np.empty((0, 4), dtype=np.float32)
        if args.only_positive_labels and (lbb.size == 0):
            continue
        tp, fp, fn, n_gt = acc(np.load(p), lbb, args.conf_th, args.nms_th, args.detect_th)
        stats['tp'] += len(tp)
        stats['fp'] += len(fp)
        stats['fn'] += len(fn)
        stats['gt'] += n_gt
        stats['scans'] += 1
        used.append(base)

    sens = stats['tp']/stats['gt'] if stats['gt'] else 0.0
    fp_per_scan = stats['fp']/stats['scans'] if stats['scans'] else 0.0
    print('cases_evaluated=', stats['scans'])
    print('gt_total=', stats['gt'])
    print('tp_total=', stats['tp'])
    print('fp_total=', stats['fp'])
    print('fn_total=', stats['fn'])
    print('sensitivity={:.4f}'.format(sens))
    print('fp_per_scan={:.4f}'.format(fp_per_scan))
    if used:
        print('first_cases:', used[:10])

if __name__ == '__main__':
    main()

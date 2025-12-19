import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Estimate remaining detection time')
    parser.add_argument('--rate-min', type=float, default=1.2,
                        help='Estimated minutes per case (default: 1.2 min/case)')
    parser.add_argument('--prep', type=str, default=None,
                        help='Override preprocess_result_path (folder with *_clean.npy)')
    parser.add_argument('--out', type=str, default=None,
                        help='Override bbox result path (folder with *_pbb.npy)')
    args = parser.parse_args()

    # Resolve paths from config_submit if not overridden
    if args.prep is None or args.out is None:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from config_submit import config as config_submit
        prep = args.prep or config_submit['preprocess_result_path']
        out = args.out or os.path.join(os.path.dirname(__file__), '..', 'bbox_result')
    else:
        prep = args.prep
        out = args.out

    if not os.path.isdir(prep):
        print('[ERROR] preprocess_result_path not found:', prep)
        return 1
    if not os.path.isdir(out):
        # If out not present yet, consider 0 done
        os.makedirs(out, exist_ok=True)

    total = len([f for f in os.listdir(prep) if f.endswith('_clean.npy')])
    done = len([f for f in os.listdir(out) if f.endswith('_pbb.npy')])
    remain = max(0, total - done)

    est_min = int(round(remain * args.rate_min))
    hours = est_min // 60
    minutes = est_min % 60

    print('total={} done={} remain={}'.format(total, done, remain))
    print('rate_min_per_case={:.2f}'.format(args.rate_min))
    if hours > 0:
        print('estimated_time ≈ {} h {} min'.format(hours, minutes))
    else:
        print('estimated_time ≈ {} min'.format(minutes))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

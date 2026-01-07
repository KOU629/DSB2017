import argparse
import random
from pathlib import Path


def read_ids(p: str):
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]


def write_ids(ids, p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ids) + '\n')


def main():
    ap = argparse.ArgumentParser(description='Create a reproducible holdout split from an ID list')
    ap.add_argument('--ids-file', required=True, help='Input ID list (one ID per line)')
    ap.add_argument('--ratio', type=float, default=0.2, help='Holdout ratio (default: 0.2)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    ap.add_argument('--out-holdout', default='tools/ids_valid_holdout.txt', help='Output path for holdout IDs')
    ap.add_argument('--out-trainlike', default='tools/ids_valid_trainlike.txt', help='Output path for remaining IDs')
    args = ap.parse_args()

    ids = sorted(read_ids(args.ids_file))
    rng = random.Random(args.seed)
    k = max(1, int(round(args.ratio * len(ids))))
    holdout = sorted(rng.sample(ids, k))
    trainlike = [i for i in ids if i not in set(holdout)]

    write_ids(holdout, args.out_holdout)
    write_ids(trainlike, args.out_trainlike)

    print(f'input_ids={len(ids)}')
    print(f'holdout_size={len(holdout)} written={args.out_holdout}')
    print(f'trainlike_size={len(trainlike)} written={args.out_trainlike}')


if __name__ == '__main__':
    main()

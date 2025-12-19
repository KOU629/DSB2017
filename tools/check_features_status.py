import argparse
import os


def read_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def main():
    ap = argparse.ArgumentParser(description='Check which cases are missing feature npy files')
    ap.add_argument('--ids-file', default='tools/ids_valid.txt')
    ap.add_argument('--bbox-dir', default='bbox_result')
    args = ap.parse_args()

    ids = read_ids(args.ids_file)
    have_feat = set([f[:-12] for f in os.listdir(args.bbox_dir) if f.endswith('_feature.npy')])
    missing = [cid for cid in ids if cid not in have_feat]

    print(f'total_ids={len(ids)}')
    print(f'features_ready={len(ids) - len(missing)}')
    print(f'missing_features={len(missing)}')
    if missing:
        print('first_missing:', missing[:20])


if __name__ == '__main__':
    main()

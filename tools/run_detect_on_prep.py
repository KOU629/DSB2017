import os
import sys
import argparse
import numpy as np
from importlib import import_module

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config_submit import config as config_submit
from split_combine import SplitComb
from data_detector import DataBowl3Detector, collate
from test_detect import test_detect

import torch
import argparse as _argparse
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Run detector on preprocessed volumes')
    parser.add_argument('--features', action='store_true', help='Also output per-proposal feature arrays')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers (default: 2)')
    parser.add_argument('--chunks-per-run', type=int, default=1, help='Split chunks processed per forward pass (default: 1)')
    parser.add_argument('--sidelen', type=int, default=144, help='SplitComb side length before margin (default: 144)')
    parser.add_argument('--margin', type=int, default=32, help='SplitComb margin (default: 32)')
    parser.add_argument('--ids', nargs='+', default=None, help='Case IDs to run (space or comma separated, e.g., 000 001 or 000,001)')
    parser.add_argument('--ids-file', type=str, default='', help='Path to a text file with one case ID per line')
    parser.add_argument('--shard-index', type=int, default=-1, help='Shard index (0-based) for partitioned runs')
    parser.add_argument('--shard-count', type=int, default=1, help='Total number of shards for partitioned runs')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of cases to run (default: 0 = all)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip cases that already have *_pbb.npy in bbox_result')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU even if CUDA is available')
    parser.add_argument('--threads', type=int, default=0, help='Set torch CPU threads (0=auto/no-change)')
    parser.add_argument('--prefetch-factor', type=int, default=1, help='DataLoader prefetch factor when workers>0 (default: 1)')
    parser.add_argument('--pbb-thresh', type=float, default=-3.0, help='Threshold for proposal selection (higher reduces proposals)')
    args = parser.parse_args()
    prep_result_path = config_submit['preprocess_result_path']
    detector_model = config_submit['detector_model']
    detector_param = config_submit['detector_param']
    n_gpu = max(1, int(config_submit.get('n_gpu', 1)))

    # 出力先（先に作っておくとスキップ判定に使える）
    bbox_result_path = os.path.join(os.path.dirname(__file__), '..', 'bbox_result')
    if not os.path.exists(bbox_result_path):
        os.makedirs(bbox_result_path)

    # 事前条件: prep_result_path に *_clean.npy が存在
    ids = [f.split('_clean.npy')[0] for f in os.listdir(prep_result_path) if f.endswith('_clean.npy')]
    # フィルタリング（IDs指定 or 個数制限）
    if args.ids:
        want = set()
        for tok in args.ids:
            for part in tok.split(','):
                part = part.strip()
                if part:
                    want.add(part)
        ids = [x for x in ids if x in want]
    if args.ids_file:
        try:
            with open(args.ids_file, 'r') as f:
                file_ids = [line.strip() for line in f if line.strip()]
            want = set(file_ids)
            ids = [x for x in ids if x in want]
        except Exception as e:
            print('[WARN] Failed to read ids-file:', e)
    # Sharding (after explicit filtering)
    if args.shard_count and args.shard_count > 1 and args.shard_index >= 0:
        shard_ids = []
        for idx, cid in enumerate(sorted(ids)):
            if idx % args.shard_count == args.shard_index:
                shard_ids.append(cid)
        ids = shard_ids
    if args.limit and args.limit > 0:
        ids = ids[:args.limit]
    if args.skip_existing:
        if args.features:
            have_pbb = set([f.split('_pbb.npy')[0] for f in os.listdir(bbox_result_path) if f.endswith('_pbb.npy')])
            have_feat = set([f.split('_feature.npy')[0] for f in os.listdir(bbox_result_path) if f.endswith('_feature.npy')])
            done = have_pbb.intersection(have_feat)
        else:
            done = set([f.split('_pbb.npy')[0] for f in os.listdir(bbox_result_path) if f.endswith('_pbb.npy')])
        ids = [x for x in ids if x not in done]
    if not ids:
        print('[ERROR] No cases to run after filtering. Checked base path: {}'.format(prep_result_path))
        print('        Possible reasons: --ids not found, --skip-existing filtered all, or sharding/limit left none.')
        sys.exit(1)

    # モデルと設定の取得
    nodmodel = import_module(detector_model.split('.py')[0])
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    config1['datadir'] = prep_result_path
    # 特徴量出力と一度に処理するチャンク数
    config1['output_feature'] = bool(args.features)
    config1['chunks_per_run'] = int(args.chunks_per_run)
    config1['pbb_thresh'] = float(args.pbb_thresh)

    # Torch CPU threading (optional)
    if args.threads and args.threads > 0:
        try:
            torch.set_num_threads(args.threads)
            # keep interop threads moderate
            torch.set_num_interop_threads(max(1, min(8, args.threads)))
            print('[INFO] Set torch threads:', args.threads)
        except Exception as e:
            print('[WARN] Failed to set torch threads:', e)

    # Load checkpoint with CPU/GPU compatibility and handle DataParallel prefixes
    def _load_ckpt_state_dict(model, ckpt):
        state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        if isinstance(state, dict):
            # strip 'module.' prefix if present
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
            model.load_state_dict(state)
        else:
            # fallback: try direct
            model.load_state_dict(ckpt)

    def _torch_load_safe(path, map_location=None, prefer_weights_only=True):
        if prefer_weights_only:
            try:
                # Allowlist argparse.Namespace when using weights_only
                from torch.serialization import safe_globals
                with safe_globals([_argparse.Namespace]):
                    return torch.load(path, map_location=map_location, weights_only=True)
            except Exception as e:
                print('[WARN] weights_only load failed, falling back to full load:', e)
        # Final fallback (only use with trusted checkpoints)
        return torch.load(path, map_location=map_location)

    use_cuda = (torch.cuda.is_available() and (not args.force_cpu))
    if use_cuda:
        # robust torch.load with safe allowlist then full fallback
        checkpoint = _torch_load_safe(detector_param, map_location=None, prefer_weights_only=True)
        _load_ckpt_state_dict(nod_net, checkpoint)
        torch.cuda.set_device(0)
        # Set feature return flag prior to DataParallel wrapping
        setattr(nod_net, 'return_feature', bool(config1['output_feature']))
        nod_net = nod_net.cuda()
        cudnn.benchmark = True
        nod_net = DataParallel(nod_net)
        print('[INFO] Using CUDA device: 0')
    else:
        checkpoint = _torch_load_safe(detector_param, map_location='cpu', prefer_weights_only=True)
        _load_ckpt_state_dict(nod_net, checkpoint)
        setattr(nod_net, 'return_feature', bool(config1['output_feature']))
        print('[INFO] Using CPU (CUDA unavailable or forced)')

    # データローダ
    margin = int(args.margin)
    sidelen = int(args.sidelen)
    split_comber = SplitComb(sidelen, config1['max_stride'], config1['stride'], margin, pad_value=config1['pad_value'])
    dataset = DataBowl3Detector(ids, config1, phase='test', split_comber=split_comber)
    pin = use_cuda
    if args.workers and args.workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin,
            prefetch_factor=max(1, int(args.prefetch_factor)),
            collate_fn=collate,
        )
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)

    # Ensure test_detect sees correct device usage: n_gpu>0 only when actually using CUDA
    n_gpu_effective = 1 if use_cuda else 0
    test_detect(loader, nod_net, get_pbb, bbox_result_path, config1, n_gpu=n_gpu_effective)
    print('[INFO] Detection done. PBB and{} feature npy saved to {}.'.format(
        '' if config1['output_feature'] else ' (no)', bbox_result_path))


if __name__ == '__main__':
    main()

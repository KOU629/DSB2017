# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

from config_training import config
from prepare import prepare_luna, preprocess_luna

if __name__ == '__main__':
    # Prepare only LUNA16 (skip Stage1)
    prepare_luna()

    seg_dir = config.get('luna_segment')
    has_seg = seg_dir and os.path.isdir(seg_dir) and any(f.endswith('.mhd') for f in os.listdir(seg_dir))
    if not has_seg:
        print('[INFO] LUNA16 segmentation not found in {}.'.format(seg_dir))
        print('       Download seg-lungs-LUNA16 and place .mhd/.zraw there, then rerun.')
        sys.exit(0)

    preprocess_luna()

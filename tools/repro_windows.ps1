param(
  [string]$PreprocessDir = "C:\Data\LUNA16\preprocess",
  [string]$RepoDir = "C:\Users\user\Documents\GitHub\DSB2017",
  [string]$Env = "dsb310"
)

# Activate environment and move to repo
conda activate $Env
Push-Location $RepoDir

# Build valid-ID list (positive-label cases)
python tools\list_valid_cases.py --preprocess-dir $PreprocessDir --out-file tools\ids_valid.txt

# Detection on all cases
python tools\run_detect_on_prep.py \
  --preprocess-dir $PreprocessDir \
  --bbox-dir bbox_result \
  --ids-file tools\all_ids.txt \
  --conf-th -0.8 --detect-th 0.35 --nms-th 0.1 \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu

# Feature extraction on positive cases
python tools\run_detect_on_prep.py \
  --preprocess-dir $PreprocessDir \
  --bbox-dir bbox_result \
  --ids-file tools\ids_valid.txt \
  --conf-th -0.8 --detect-th 0.35 --nms-th 0.1 \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu --features

# Evaluation: sensitivity (positives) and FP/scan (all)
python tools\eval_pbb.py --ids-file tools\ids_valid.txt --only-positive-labels --conf-th -0.8 --nms-th 0.1 --detect-th 0.35
python tools\eval_pbb.py --ids-file tools\all_ids.txt --conf-th -0.8 --nms-th 0.1 --detect-th 0.35

# Feature probe (purity & AUC)
python tools\eval_features_probe.py --ids-file tools\ids_valid.txt --bbox-dir bbox_result --detect-th 0.35 --probe

Pop-Location

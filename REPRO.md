# DSB2017 再現バンドル（Windows/PowerShell）

このガイドは、LUNA16の前処理済みボリュームで「検出」と「特徴抽出」を実行し、評価まで再現するための最小手順をまとめたものです。コマンドは Windows PowerShell 用です。

## 前提条件
- Conda 環境: Python 3.10（例: dsb310）
- 必要パッケージ: torch（CPU版）、numpy、scipy、scikit-image
- 前処理ディレクトリ: 例 `C:\Data\LUNA16\preprocess`（準備済み）
- リポジトリ配置: `C:\Users\user\Documents\GitHub\DSB2017`

## 主要スクリプト
- 検出実行: [tools/run_detect_on_prep.py](tools/run_detect_on_prep.py)
- 有効症例リスト作成: [tools/list_valid_cases.py](tools/list_valid_cases.py)
- 評価: [tools/eval_pbb.py](tools/eval_pbb.py)
- 特徴プローブ: [tools/eval_features_probe.py](tools/eval_features_probe.py)
- オプション（ホールドアウト分割）: [tools/make_holdout.py](tools/make_holdout.py)

## クイックスタート（コピー&ペースト）
```powershell
# 1) 環境を有効化してリポジトリへ移動
conda activate dsb310
Push-Location "C:\Users\user\Documents\GitHub\DSB2017"

# 2) 前処理ディレクトリから有効ID（陽性症例）を作成
python tools\list_valid_cases.py --preprocess-dir "C:\Data\LUNA16\preprocess" --out-file tools\ids_valid.txt

# 3) 検出（全症例）: *_pbb.npy と *_lbb.npy を生成
python tools\run_detect_on_prep.py \
  --preprocess-dir "C:\Data\LUNA16\preprocess" \
  --bbox-dir bbox_result \
  --ids-file tools\all_ids.txt \
  --conf-th -0.8 --detect-th 0.35 --nms-th 0.1 \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu

# 4) 特徴抽出（陽性症例）: *_feature.npy を追加生成
python tools\run_detect_on_prep.py \
  --preprocess-dir "C:\Data\LUNA16\preprocess" \
  --bbox-dir bbox_result \
  --ids-file tools\ids_valid.txt \
  --conf-th -0.8 --detect-th 0.35 --nms-th 0.1 \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu --features

# 5) 評価（陽性のみ感度 / 全症例のFP/scan）
python tools\eval_pbb.py --ids-file tools\ids_valid.txt --only-positive-labels --conf-th -0.8 --nms-th 0.1 --detect-th 0.35
python tools\eval_pbb.py --ids-file tools\all_ids.txt --conf-th -0.8 --nms-th 0.1 --detect-th 0.35

# 6) 特徴品質（提案純度 & プローブAUC）
python tools\eval_features_probe.py --ids-file tools\ids_valid.txt --bbox-dir bbox_result --detect-th 0.35 --probe

Pop-Location
```

## 期待される出力
- [bbox_result/](bbox_result) に検出結果が保存されます:
  - `*_pbb.npy`: 提案ボックスと信頼度
  - `*_lbb.npy`: 正解ラベル（存在する症例のみ）
  - `*_feature.npy`: 128次元の特徴ベクトル（`--features` 指定時）

## 推奨閾値
- `conf_th = -0.8`, `detect_th = 0.35`, `nms_th = 0.1`
- 参考値（過去実測）: 感度 ≈ 0.937（陽性症例）、FP/scan ≈ 4.55（全症例）、特徴プローブAUC ≈ 0.895

## オプション: ホールドアウト分割
```powershell
python tools\make_holdout.py --ids-file tools\ids_valid.txt --ratio 0.2 --seed 42 --out-holdout tools\ids_valid_holdout.txt --out-trainlike tools\ids_valid_trainlike.txt
python tools\eval_pbb.py --ids-file tools\ids_valid_holdout.txt --only-positive-labels --conf-th -0.8 --nms-th 0.1 --detect-th 0.35
```

## 注意・ヒント
- PowerShell で連結する場合は `;` を使用（可読性のため行ごと推奨）。
- ここでは CPU 実行を想定。GPU が非対応の場合は `--force-cpu` を維持。
- 評価スクリプトは直径 ≤ 0 の無効ラベルを自動で除外します。
- FP/scan を下げたい場合は `detect_th` を上げる（例: 0.40/0.50）→再評価。

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

# 0) 前処理済みデータの場所（*_clean.npy と *_label.npy があるディレクトリ）
$PREP = "C:\Data\LUNA16\preprocess"
$BBOX = "bbox_result"

# NOTE: LUNA16のGT評価をするには、$PREP に *_label.npy（直径>0のラベル）が必要です。
#       これが無い場合、検出評価（sensitivity/FP）は意味のある値になりません。

# 2) 検出（全症例）: *_pbb.npy と *_lbb.npy を生成
python tools\run_detect_on_prep.py \
  --preprocess-dir "$PREP" \
  --bbox-dir "$BBOX" \
  --ids-file tools\all_ids.txt \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu

# 3) 有効ID（直径>0のGTが存在する症例）を作成
python tools\list_valid_cases.py --bbox-dir "$BBOX" --out tools\ids_valid.txt --min-diameter 0

# 4) 特徴抽出（陽性症例）: *_feature.npy を追加生成
python tools\run_detect_on_prep.py \
  --preprocess-dir "$PREP" \
  --bbox-dir "$BBOX" \
  --ids-file tools\ids_valid.txt \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu --features

# (任意) top-K の特徴・座標を追加保存: *_feat.npy と *_coord.npy
# - *_feat.npy は (K, 128) の検出器特徴（proposalごと）
# - *_coord.npy は (K, 5) の [conf,z,y,x,d]
python tools\run_detect_on_prep.py \
  --preprocess-dir "$PREP" \
  --bbox-dir "$BBOX" \
  --ids-file tools\ids_valid.txt \
  --skip-existing --workers 2 --chunks-per-run 64 --sidelen 144 --margin 16 \
  --force-cpu --features --save-topk --topk 5

# 5) 評価（陽性のみ感度 / 全症例のFP/scan）
python tools\eval_pbb.py --ids-file tools\ids_valid.txt --only-positive-labels --conf-th -0.8 --nms-th 0.1 --detect-th 0.35
python tools\eval_pbb.py --ids-file tools\all_ids.txt --conf-th -0.8 --nms-th 0.1 --detect-th 0.35

# 6) 特徴品質（提案純度 & プローブAUC）
python tools\eval_features_probe.py --ids-file tools\ids_valid.txt --bbox-dir "$BBOX" --detect-th 0.35 --probe

Pop-Location
```

## 期待される出力
- [bbox_result/](bbox_result) に検出結果が保存されます:
  - `*_pbb.npy`: 提案ボックスと信頼度
  - `*_lbb.npy`: 正解ラベル（存在する症例のみ）
  - `*_feature.npy`: 128次元の特徴ベクトル（`--features` 指定時、proposalごとに1本）
  - `*_feat.npy`: top-K proposalの特徴（`--save-topk --topk K` 指定時）
  - `*_coord.npy`: top-K proposalの [conf,z,y,x,d]（`--save-topk --topk K` 指定時）

## 特徴とproposalの対応について
- `*_feature.npy` は `*_pbb.npy` の行と 1:1 に対応することを想定しています（同一runの出力同士で対応）。
- 後段で `*_pbb.npy` に NMS/閾値処理を追加適用する場合、同じインデックスで `*_feature.npy` 側も同様に間引いてください。

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

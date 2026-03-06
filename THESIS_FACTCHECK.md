# Thesis fact-check (DSB2017 repo)

このメモは、このリポジトリ（DSB2017）の実装を根拠に、卒論本文で誤解が起きやすい点・断定が危険な点を整理したものです。

## 1) 前処理（LUNA16想定）
- 実装根拠: preprocessing/full_prep.py
- HUウィンドウ: [-1200, 600]
- 0–1正規化 → 0–255へスケールし、uint8に変換（8bit量子化）
- 肺野マスク（凸包 + 膨張）で肺外をpad_value=170で埋める、骨（bone_thresh=210）もpad_valueへ
- 等方化: [1,1,1]mmにリサンプリング（order=1）
- 出力: *_clean.npy（先頭にチャネル次元が付く）

注意:
- LUNA16の評価を行うには、*_label.npy に「直径>0のGT」が入っている必要があります（full_prep.py はGTを生成しません）。
  - GT生成を含むLUNA前処理は training/prepare.py の preprocess_luna() が担当します。

## 2) 検出器の出力（pbb）
- 実装根拠: net_detector.py / layers.py:GetPBB
- anchors=[10,30,60]mm, stride=4
- 出力はグリッド×アンカーごとに [conf, dz, dy, dx, dd] を持ち、GetPBBで [conf,z,y,x,d] へデコード
- 推論時の保存: bbox_result/*_pbb.npy, bbox_result/*_lbb.npy

## 3) 特徴抽出（このリポジトリで「実際に保存される特徴」）
- 実装根拠: test_detect.py / net_detector.py
- tools/run_detect_on_prep.py に --features を付けると、検出器の中間特徴 feat（チャネル数=128）をproposalごとに保存します。
  - 保存先: bbox_result/*_feature.npy
  - 形状: (N, 128)（Nはpbb_threshで残ったproposal数）
  - 重要: *_feature.npy は「検出器由来のproposal特徴」です。CaseNet（分類器）の特徴ではありません。

## 4) top-K 形式（時系列解析向け）
- 追加実装: test_detect.py + tools/run_detect_on_prep.py
- --save-topk --topk K を付けると、confでソートしたtop-Kを以下で保存します:
  - bbox_result/*_feat.npy : (K, 128) detector feature
  - bbox_result/*_coord.npy : (K, 5) = [conf,z,y,x,d]

## 5) 評価指標と公式FROCとの差
- 実装根拠: tools/eval_pbb.py → layers.acc
- TP判定は3D IoU（detect_th）に基づきます。
- LUNA16公式の「中心距離ベースのFROC」とは一致しないため、本文で比較する場合は注記が必要です。

## 6) 小結節（サイズ閾値）の扱い
- 実装根拠: data_detector.py（train/val）
- 学習用サンプリングでは、直径が sizelim(=6mm) を超えるもののみをbbox中心として扱います（小結節はサンプル中心にならない）。
- 一方、評価（tools/eval_pbb.py）側は「直径<=0の無効ラベル」を除外するだけです。

---

必要なら、このメモに合わせて「卒論側の章ごとの修正案（文章案）」もこのリポジトリ内にまとめられます（ただし、卒論texがこのワークスペース外だと直接パッチ適用はできません）。

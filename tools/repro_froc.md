# FROC/CPM 再現手順

## 1. 評価に使うID集合
- 推奨: [tools/all_ids.txt](tools/all_ids.txt)（陽性+陰性）
- 代替: [tools/ids_valid.txt](tools/ids_valid.txt)（陽性のみ。FP/scanの解釈に注意）

## 1.1 TP/FP/FN の定義（マッチング規則の明文化）
このリポジトリの検出評価は、[layers.py](layers.py) の `acc()` / `iou()` の実装に従います。

- 予測候補（pbb）とGT（lbb）は、いずれも `[z, y, x, d]`（中心座標 + 直径）として扱われます。
- `iou()` は、中心 `(z,y,x)`、辺長 `d` の **軸平行立方体（cube）** 同士の 3D IoU を計算します。
  - 半径 `r=d/2`、立方体の範囲は `center±r`
  - 体積は `d^3`
- `detect_th` は「TPとみなす最小IoU」で、評価プロトコルの固定パラメータです（FROCではスイープしない）。
- 1つのGTに複数の候補がマッチした場合、最初の1つのみをTP、2つ目以降は **重複検出としてFP** に計上します。
- どのGTにもマッチしない候補はFP、どの候補にもマッチしないGTはFNです。

## 2. しきい値スイープ（評価CSVを作成）
- NMS閾値: 0.1（固定）
- TP判定基準（マッチング規則）: detect_th（固定。FROCではスイープしない）
- only-positive-labels: 付ける/付けないを固定
- 予測/GTファイルの場所: --bbox-dir（デフォルト: bbox_result）
- しきい値範囲（例）
  - conf_th: -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
  - 低FP/scan領域（<1）に到達しない場合は conf_th をさらに高い方へ拡張

PowerShell例（陽性+陰性、only-positive-labelsなし）:
Push-Location "c:\Users\user\Documents\GitHub\DSB2017"
python tools/sweep_froc.py --ids-file tools/all_ids.txt --bbox-dir bbox_result --nms-th 0.1 --detect-th 0.1 --conf-th -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
Pop-Location

## 3. FROC曲線/CPMの生成
- 入力: sweep_froc_results.csv（または sweep_eval_results.csv 互換のCSV）
- 出力: froc.png, froc_points.csv, froc_summary.json, froc_table.csv, froc_table.md

PowerShell例:
Push-Location "c:\Users\user\Documents\GitHub\DSB2017"
python tools/plot_froc_cpm.py --in-csv tools/sweep_froc_results.csv --label baseline --ids-file tools/all_ids.txt --nms-th 0.1 --sweep-conf -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
Pop-Location

## 4. 改善前/改善後を同一図に重ねる
- 2回スイープしてCSVを分ける（例: sweep_froc_results_baseline.csv, sweep_froc_results_improved.csv）

PowerShell例:
Push-Location "c:\Users\user\Documents\GitHub\DSB2017"
python tools/plot_froc_cpm.py --in-csv tools/sweep_froc_results_baseline.csv tools/sweep_froc_results_improved.csv --label baseline improved --out-png tools/froc.png --out-csv tools/froc_points.csv --out-json tools/froc_summary.json --ids-file tools/all_ids.txt --nms-th 0.1 --sweep-conf -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
Pop-Location

## 5. 表に載せる値
- CPM: tools/froc_summary.json の runs[*].cpm
- 運用点（conf_th, detect_th）: --op-point で指定
  - 出力: runs[*].operating_metrics に TP/FP/FN/Sensitivity/FP/scan が含まれる

例:
python tools/plot_froc_cpm.py --in-csv tools/sweep_froc_results_baseline.csv tools/sweep_froc_results_improved.csv --label baseline improved --op-point -0.8 0.1 -1.2 0.1

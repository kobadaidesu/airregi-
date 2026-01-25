# 高負荷日アラート評価システム

コールセンター入電数（call_num）の翌営業日予測モデルに基づく「高負荷日アラート」の実装と評価。

## 実行方法

```bash
cd kaggle-template
python run_alert_eval.py
```

## 出力ファイル

出力先: `output/alert_eval/`

| ファイル名 | 説明 |
|-----------|------|
| `metrics_forecast.json` | 予測精度指標（MAE, WAPE） |
| `metrics_alert.json` | アラート精度指標（Precision, Recall, F1, Confusion Matrix） |
| `simulation_summary.csv` | モデル別のシミュレーション結果サマリー |
| `simulation_daily_detail.csv` | 日次の詳細データ |
| `slide_ready_summary.md` | スライドに貼れる短文サマリー |
| `actual_vs_predicted.png` | 実績 vs 予測グラフ |
| `alert_threshold.png` | アラート閾値と当たり外れ |
| `cumulative_cost.png` | モデル別累積コスト推移 |

## 設定パラメータ

スクリプト内で以下のパラメータを調整可能：

```python
# 運用シミュレーションパラメータ
CAP = 60           # 1人あたり対応可能コール数/日
BUFFER = 1.1       # 安全余裕係数
C_SHORT = 600      # 不足1コールのコスト（外注/残業/機会損失の代理）
C_OVER = 50        # 過剰1コールのコスト（ムダ配置の代理）

# Holdout検証設定
VALIDATION_DAYS = 60  # 検証期間（末尾N日）

# アラート閾値設定
USE_DYNAMIC_THRESHOLD = True  # 動的閾値モード
HIGH_LOAD_PERCENTILE = 80     # 上位X%を高負荷日とする
```

## モデル

- **CatBoost**: exp22で最適化されたパラメータを使用
- **lag7**: 前週同曜日の値（フォールバック付き）
- **ma7**: 直近7日移動平均

## 注意事項

1. アラートは予測の"補助"であり、日次の要員計画は予測値そのものを使用
2. c_short/c_over/cap/buffer は代理パラメータで、感度分析の余地がある
3. 検証期間（2020年1-3月）はCOVID-19影響で入電数が全体的に低下していた可能性あり

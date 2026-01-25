#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Airレジ ヘルプデスク入電数予測 - オンボーディング負荷分析
============================================================
目的: オンボーディング（acc_get_cnt等）起因の入電量を推定し、
      AI Agent導入による削減可能コール数を定量評価する

時系列リーク防止:
  - 学習: 2018-06-01 ~ 2020-01-29
  - 検証: 2020-01-30 ~ 2020-03-30 (データ終端)
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "input" / "airregi_features_verified_safe.csv"
OUTPUT_DIR = BASE_DIR / "output" / "onboarding_analysis"

# 時系列分割
TRAIN_END = "2020-01-29"
VAL_START = "2020-01-30"
VAL_END = "2020-03-30"

# AI Agent 自己解決率シナリオ
RESOLUTION_RATES = [0.10, 0.20, 0.30]

# acc系特徴量のパターン（除去対象）
ACC_PATTERNS = ['acc_get_cnt', 'acc_ma', 'acc_lag', 'acc_7dma', 'acc_diff',
                'acc_squared', 'acc_x_', 'acc_cumsum', 'acc_weighted', 'acc_sum',
                'acc_high_flag', 'cumulative_acc']


def load_data():
    """データ読み込み"""
    print("=" * 60)
    print("1. データ読み込み")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    df['cdr_date'] = pd.to_datetime(df['cdr_date'])
    df = df.sort_values('cdr_date').reset_index(drop=True)

    print(f"  データ期間: {df['cdr_date'].min().date()} ~ {df['cdr_date'].max().date()}")
    print(f"  総レコード数: {len(df)}")
    print(f"  カラム数: {len(df.columns)}")

    return df


def identify_acc_features(columns):
    """acc系特徴量を特定"""
    acc_cols = []
    for col in columns:
        col_lower = col.lower()
        for pattern in ACC_PATTERNS:
            if pattern.lower() in col_lower:
                acc_cols.append(col)
                break
    return list(set(acc_cols))


def prepare_features(df):
    """特徴量準備"""
    print("\n" + "=" * 60)
    print("2. 特徴量準備")
    print("=" * 60)

    # 目的変数
    target = 'call_num'

    # 除外カラム
    exclude_cols = ['cdr_date', 'call_num', 'dow_name', 'holiday_name']

    # 全特徴量（数値型のみ）
    all_feature_cols = [c for c in df.columns
                        if c not in exclude_cols
                        and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]

    # acc系特徴量を特定
    acc_features = identify_acc_features(all_feature_cols)

    # NoAcc用特徴量
    noacc_feature_cols = [c for c in all_feature_cols if c not in acc_features]

    print(f"  Full モデル特徴量数: {len(all_feature_cols)}")
    print(f"  acc系特徴量数: {len(acc_features)}")
    print(f"  NoAcc モデル特徴量数: {len(noacc_feature_cols)}")
    print(f"\n  acc系特徴量一覧:")
    for f in sorted(acc_features):
        print(f"    - {f}")

    return all_feature_cols, noacc_feature_cols, acc_features, target


def split_data(df, all_features, noacc_features, target):
    """時系列分割（リーク防止）"""
    print("\n" + "=" * 60)
    print("3. 時系列分割")
    print("=" * 60)

    # 学習データ
    train_mask = df['cdr_date'] <= TRAIN_END
    train_df = df[train_mask].copy()

    # 検証データ
    val_mask = (df['cdr_date'] >= VAL_START) & (df['cdr_date'] <= VAL_END)
    val_df = df[val_mask].copy()

    print(f"  学習期間: ~ {TRAIN_END} ({len(train_df)} 件)")
    print(f"  検証期間: {VAL_START} ~ {VAL_END} ({len(val_df)} 件)")

    # 休日・欠損の扱い
    # 休日（call_num=0 かつ holiday_flag=True）は検証から除外しない（実態を反映）
    # ただし call_num=0 の日は注意書きに明記
    n_zero_call = (val_df['call_num'] == 0).sum()
    print(f"  検証期間内の call_num=0 日数: {n_zero_call}")

    # 欠損除去
    train_df = train_df.dropna(subset=all_features + [target])
    val_df = val_df.dropna(subset=all_features + [target])

    print(f"  学習データ（欠損除去後）: {len(train_df)} 件")
    print(f"  検証データ（欠損除去後）: {len(val_df)} 件")

    return train_df, val_df


def train_model(X_train, y_train, model_name="model"):
    """CatBoostモデル学習"""
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """モデル評価"""
    y_pred = model.predict(X_val)

    # MAE
    mae = np.mean(np.abs(y_val - y_pred))

    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(y_val - y_pred)) / np.sum(np.abs(y_val)) * 100

    return y_pred, mae, wape


def compute_shap_values(model, X_val, feature_names):
    """SHAP値を計算"""
    print("\n  SHAP値を計算中...")

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        # DataFrameに変換
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        return shap_df
    except ImportError:
        print("  警告: shapライブラリが見つかりません。SHAP分析をスキップします。")
        return None
    except Exception as e:
        print(f"  警告: SHAP計算でエラー: {e}")
        return None


def run_analysis():
    """メイン分析"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    df = load_data()

    # 特徴量準備
    all_features, noacc_features, acc_features, target = prepare_features(df)

    # 時系列分割
    train_df, val_df = split_data(df, all_features, noacc_features, target)

    # =============================================================================
    # モデル学習
    # =============================================================================
    print("\n" + "=" * 60)
    print("4. モデル学習")
    print("=" * 60)

    X_train_full = train_df[all_features]
    X_train_noacc = train_df[noacc_features]
    y_train = train_df[target]

    X_val_full = val_df[all_features]
    X_val_noacc = val_df[noacc_features]
    y_val = val_df[target].values

    print("  Full モデル学習中...")
    model_full = train_model(X_train_full, y_train, "Full")

    print("  NoAcc モデル学習中...")
    model_noacc = train_model(X_train_noacc, y_train, "NoAcc")

    # =============================================================================
    # モデル評価
    # =============================================================================
    print("\n" + "=" * 60)
    print("5. モデル評価")
    print("=" * 60)

    pred_full, mae_full, wape_full = evaluate_model(model_full, X_val_full, y_val)
    pred_noacc, mae_noacc, wape_noacc = evaluate_model(model_noacc, X_val_noacc, y_val)

    print(f"  Full モデル:  MAE = {mae_full:.2f}, WAPE = {wape_full:.2f}%")
    print(f"  NoAcc モデル: MAE = {mae_noacc:.2f}, WAPE = {wape_noacc:.2f}%")

    # =============================================================================
    # アブレーション評価
    # =============================================================================
    print("\n" + "=" * 60)
    print("6. アブレーション評価")
    print("=" * 60)

    delta = pred_full - pred_noacc

    ablation_df = pd.DataFrame({
        'date': val_df['cdr_date'].values,
        'actual': y_val,
        'pred_full': pred_full,
        'pred_noacc': pred_noacc,
        'delta': delta
    })

    # 基本統計
    delta_sum = delta.sum()
    delta_mean = delta.mean()
    delta_positive_sum = delta[delta > 0].sum()

    print(f"  Δ (pred_full - pred_noacc) の統計:")
    print(f"    合計: {delta_sum:.2f}")
    print(f"    平均: {delta_mean:.2f}")
    print(f"    正値のみ合計: {delta_positive_sum:.2f}")

    # 上位日（ピーク）
    top_days = ablation_df.nlargest(10, 'delta')[['date', 'actual', 'pred_full', 'pred_noacc', 'delta']]
    print(f"\n  Δ上位10日（ピーク日）:")
    for _, row in top_days.iterrows():
        print(f"    {row['date'].date()}: Δ={row['delta']:.1f} (実績={row['actual']:.0f})")

    # =============================================================================
    # SHAP分析
    # =============================================================================
    print("\n" + "=" * 60)
    print("7. SHAP分析")
    print("=" * 60)

    shap_df = compute_shap_values(model_full, X_val_full, all_features)

    shap_acc_daily = None
    shap_acc_positive_total = 0

    if shap_df is not None:
        # acc系特徴量のSHAP値を日次合計
        acc_shap_cols = [c for c in shap_df.columns if c in acc_features]

        if acc_shap_cols:
            shap_acc_daily = shap_df[acc_shap_cols].sum(axis=1).values
            shap_acc_positive = np.maximum(shap_acc_daily, 0)
            shap_acc_positive_total = shap_acc_positive.sum()

            print(f"  acc系特徴量のSHAP寄与:")
            print(f"    合計（全体）: {shap_acc_daily.sum():.2f}")
            print(f"    合計（正のみ）: {shap_acc_positive_total:.2f}")
            print(f"    平均: {shap_acc_daily.mean():.2f}")

            ablation_df['shap_acc_contribution'] = shap_acc_daily

    # =============================================================================
    # AI Agent効果シナリオ
    # =============================================================================
    print("\n" + "=" * 60)
    print("8. AI Agent効果シナリオ")
    print("=" * 60)

    # 推定オンボーディング寄与入電量
    # 方法1: アブレーションのΔ（正値のみ）
    onboarding_calls_ablation = delta_positive_sum

    # 方法2: SHAP（正値のみ）
    onboarding_calls_shap = shap_acc_positive_total if shap_acc_daily is not None else 0

    # 両方の推定値を使用
    print(f"  推定オンボーディング寄与入電量:")
    print(f"    アブレーション法: {onboarding_calls_ablation:.1f} コール")
    print(f"    SHAP法: {onboarding_calls_shap:.1f} コール")

    scenarios = []
    for r in RESOLUTION_RATES:
        reduction_ablation = r * onboarding_calls_ablation
        reduction_shap = r * onboarding_calls_shap
        scenarios.append({
            'resolution_rate': r,
            'reduction_ablation': reduction_ablation,
            'reduction_shap': reduction_shap
        })
        print(f"\n    自己解決率 {r*100:.0f}%:")
        print(f"      削減コール数（アブレーション）: {reduction_ablation:.1f}")
        print(f"      削減コール数（SHAP）: {reduction_shap:.1f}")

    scenarios_df = pd.DataFrame(scenarios)

    # =============================================================================
    # 出力
    # =============================================================================
    print("\n" + "=" * 60)
    print("9. 出力ファイル生成")
    print("=" * 60)

    # metrics.json
    metrics = {
        'full_model': {'MAE': round(mae_full, 2), 'WAPE': round(wape_full, 2)},
        'noacc_model': {'MAE': round(mae_noacc, 2), 'WAPE': round(wape_noacc, 2)},
        'ablation': {
            'delta_sum': round(delta_sum, 2),
            'delta_mean': round(delta_mean, 2),
            'delta_positive_sum': round(delta_positive_sum, 2)
        },
        'shap': {
            'acc_contribution_total': round(shap_acc_daily.sum(), 2) if shap_acc_daily is not None else None,
            'acc_contribution_positive_total': round(shap_acc_positive_total, 2) if shap_acc_daily is not None else None
        },
        'validation_period': {
            'start': VAL_START,
            'end': VAL_END,
            'n_days': len(val_df),
            'n_zero_call_days': int((val_df['call_num'] == 0).sum())
        }
    }

    with open(OUTPUT_DIR / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  metrics.json 保存完了")

    # ablation_daily.csv
    ablation_df.to_csv(OUTPUT_DIR / "ablation_daily.csv", index=False)
    print(f"  ablation_daily.csv 保存完了")

    # impact_scenarios.csv
    scenarios_df.to_csv(OUTPUT_DIR / "impact_scenarios.csv", index=False)
    print(f"  impact_scenarios.csv 保存完了")

    # =============================================================================
    # 図の生成
    # =============================================================================
    print("\n" + "=" * 60)
    print("10. 図の生成")
    print("=" * 60)

    # 図1: 実績 vs 予測
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(ablation_df['date'], ablation_df['actual'], label='実績', alpha=0.8)
    ax1.plot(ablation_df['date'], ablation_df['pred_full'], label='予測 (Full)', alpha=0.8)
    ax1.set_xlabel('日付')
    ax1.set_ylabel('入電数')
    ax1.set_title(f'検証期間の実績 vs 予測 (MAE={mae_full:.1f}, WAPE={wape_full:.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / "fig_actual_vs_pred.png", dpi=150)
    plt.close(fig1)
    print(f"  fig_actual_vs_pred.png 保存完了")

    # 図2: Δ時系列
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = ['green' if d >= 0 else 'red' for d in ablation_df['delta']]
    ax2.bar(ablation_df['date'], ablation_df['delta'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=delta_mean, color='blue', linestyle='--', linewidth=1, label=f'平均={delta_mean:.1f}')
    ax2.set_xlabel('日付')
    ax2.set_ylabel('Δ (Full - NoAcc)')
    ax2.set_title('アブレーション: acc系特徴量による予測差分')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "fig_delta_timeseries.png", dpi=150)
    plt.close(fig2)
    print(f"  fig_delta_timeseries.png 保存完了")

    # 図3: シナリオ棒グラフ
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(RESOLUTION_RATES))
    width = 0.35

    bars1 = ax3.bar(x - width/2, scenarios_df['reduction_ablation'], width,
                    label='アブレーション法', color='steelblue')
    bars2 = ax3.bar(x + width/2, scenarios_df['reduction_shap'], width,
                    label='SHAP法', color='coral')

    ax3.set_xlabel('AI Agent 自己解決率')
    ax3.set_ylabel('削減コール数')
    ax3.set_title('AI Agent導入による削減コール数シナリオ\n（検証期間中の推定）')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{r*100:.0f}%' for r in RESOLUTION_RATES])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 数値ラベル
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "fig_scenario_bar.png", dpi=150)
    plt.close(fig3)
    print(f"  fig_scenario_bar.png 保存完了")

    # =============================================================================
    # スライド用サマリ
    # =============================================================================
    print("\n" + "=" * 60)
    print("11. サマリ生成")
    print("=" * 60)

    total_actual = ablation_df['actual'].sum()
    pct_of_total = (delta_positive_sum / total_actual * 100) if total_actual > 0 else 0

    summary = f"""# オンボーディング負荷による入電量推定と AI Agent 効果試算

## 結論
- 検証期間（{VAL_START} ～ {VAL_END}）において、**オンボーディング（新規獲得）起因の入電量は約 {delta_positive_sum:.0f} コール**と推定される（アブレーション法）
- これは検証期間の総入電数 {total_actual:.0f} コールの約 **{pct_of_total:.1f}%** に相当
- AI Agent の自己解決率を **20%** と仮定した場合、**{0.2 * delta_positive_sum:.0f} コール削減**が見込まれる

## 主な数値
| 指標 | 値 |
|------|-----|
| Full モデル MAE | {mae_full:.1f} |
| Full モデル WAPE | {wape_full:.1f}% |
| NoAcc モデル MAE | {mae_noacc:.1f} |
| NoAcc モデル WAPE | {wape_noacc:.1f}% |
| Δ合計（正のみ） | {delta_positive_sum:.1f} コール |
| Δ平均 | {delta_mean:.2f} コール/日 |

## AI Agent 導入シナリオ
| 自己解決率 | 削減コール数 |
|------------|--------------|
| 10% | {0.1 * delta_positive_sum:.0f} コール |
| 20% | {0.2 * delta_positive_sum:.0f} コール |
| 30% | {0.3 * delta_positive_sum:.0f} コール |

## 注意事項
- 検証期間内に call_num=0 の日が **{int((val_df['call_num'] == 0).sum())} 日**存在（休日扱い）
- 休日は予測から除外せず、実態を反映した評価としている
- acc系特徴量（acc_get_cnt, acc_ma_*, acc_lag_* 等）を除去したモデルとの差分で寄与を推定
- この推定は「獲得増による入電増」の上限的な目安であり、実際の削減効果は AI Agent の性能や導入方法に依存

## 生成日時
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(OUTPUT_DIR / "slide_ready_summary.md", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  slide_ready_summary.md 保存完了")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
    print(f"出力先: {OUTPUT_DIR}")

    return metrics, ablation_df, scenarios_df


if __name__ == "__main__":
    metrics, ablation_df, scenarios_df = run_analysis()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_alert_eval.py

コールセンター入電数（call_num）の翌営業日予測モデルに基づく
「高負荷日アラート」の実装と評価スクリプト

Usage:
    python run_alert_eval.py

Output:
    output/alert_eval/ フォルダに以下を出力
    - metrics_forecast.json
    - metrics_alert.json
    - simulation_summary.csv
    - simulation_daily_detail.csv
    - slide_ready_summary.md
    - actual_vs_predicted.png
    - alert_threshold.png
    - cumulative_cost.png
"""

import os
import sys
import json
import warnings
from datetime import timedelta
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# 日本語フォント設定（利用可能な場合）
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    pass

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# =============================================================================
# 設定
# =============================================================================
OUTPUT_DIR = 'output/alert_eval'
INPUT_DIR = 'input'

# 運用シミュレーションパラメータ
CAP = 60           # 1人あたり対応可能コール数/日
BUFFER = 1.1       # 安全余裕係数
C_SHORT = 600      # 不足1コールのコスト（外注/残業/機会損失の代理）
C_OVER = 50        # 過剰1コールのコスト（ムダ配置の代理）

# Holdout検証設定
VALIDATION_DAYS = 60  # 検証期間（末尾N日）

# アラート閾値設定
USE_DYNAMIC_THRESHOLD = True  # Trueの場合、検証期間内の分布で閾値を決定（リーク注意）
HIGH_LOAD_PERCENTILE = 80  # 上位X%を高負荷日とする

# CatBoostパラメータ（exp22最適化済み）
CATBOOST_PARAMS = {
    'iterations': 1547,
    'learning_rate': 0.04313835983436318,
    'depth': 4,
    'l2_leaf_reg': 0.13964878723609409,
    'subsample': 0.8418882107293159,
    'random_state': 42,
    'verbose': 0
}


# =============================================================================
# データ読み込み・前処理関数
# =============================================================================
def find_csv_files(base_dir):
    """CSVファイルを探索"""
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def load_and_preprocess_data(input_dir):
    """データ読み込みと前処理"""
    print("\n" + "=" * 80)
    print("Step 1: データ読み込み")
    print("=" * 80)

    # CSVファイル探索
    csv_files = find_csv_files(input_dir)
    print(f"\n発見したCSVファイル:")
    for f in csv_files:
        print(f"  - {f}")

    # 各データ読み込み
    calender = pd.read_csv(os.path.join(input_dir, 'calender_data.csv'))
    cm_data = pd.read_csv(os.path.join(input_dir, 'cm_data.csv'))
    gt_service = pd.read_csv(os.path.join(input_dir, 'gt_service_name.csv'))
    acc_get = pd.read_csv(os.path.join(input_dir, 'regi_acc_get_data_transform.csv'))
    call_data = pd.read_csv(os.path.join(input_dir, 'regi_call_data_transform.csv'))

    # 日付変換
    calender['cdr_date'] = pd.to_datetime(calender['cdr_date'])
    cm_data['cdr_date'] = pd.to_datetime(cm_data['cdr_date'])
    acc_get['cdr_date'] = pd.to_datetime(acc_get['cdr_date'])
    call_data['cdr_date'] = pd.to_datetime(call_data['cdr_date'])
    gt_service['week'] = pd.to_datetime(gt_service['week'])

    print(f"\n読み込んだデータ:")
    print(f"  - call_data: {len(call_data)}行")
    print(f"  - calender: {len(calender)}行")
    print(f"  - cm_data: {len(cm_data)}行")
    print(f"  - acc_get: {len(acc_get)}行")
    print(f"  - gt_service: {len(gt_service)}行")

    # データ結合
    df = call_data.copy()
    df = df.merge(calender, on='cdr_date', how='left')
    df = df.merge(cm_data, on='cdr_date', how='left')
    df = df.merge(acc_get, on='cdr_date', how='left')

    # Google Trendsデータを日次に展開
    gt_service_daily = []
    for idx, row in gt_service.iterrows():
        week_start = row['week']
        for i in range(7):
            date = week_start + timedelta(days=i)
            gt_service_daily.append({'cdr_date': date, 'search_cnt': row['search_cnt']})
    gt_daily = pd.DataFrame(gt_service_daily)
    df = df.merge(gt_daily, on='cdr_date', how='left')

    print(f"\n結合後データ: {len(df)}行")
    print(f"\n列一覧:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    return df


def create_features(df):
    """特徴量作成（exp22ベース）"""
    print("\n" + "=" * 80)
    print("Step 2: 特徴量作成")
    print("=" * 80)

    df = df.copy()

    # 基本時系列特徴量
    df['year'] = df['cdr_date'].dt.year
    df['day_of_month'] = df['cdr_date'].dt.day
    df['day_of_year'] = df['cdr_date'].dt.dayofyear
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # ラグ特徴量
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f'lag_{lag}'] = df['call_num'].shift(lag)

    # 移動平均特徴量
    for window in [3, 7, 30]:
        df[f'ma_{window}'] = df['call_num'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'ma_std_{window}'] = df['call_num'].shift(1).rolling(window=window, min_periods=1).std()

    # 集約特徴量
    df['cm_7d'] = df['cm_flg'].shift(1).rolling(window=7, min_periods=1).sum()
    df['gt_ma_7'] = df['search_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
    df['acc_ma_7'] = df['acc_get_cnt'].shift(1).rolling(window=7, min_periods=1).mean()

    # 曜日別平均
    df['dow_avg'] = np.nan
    for dow in df['dow'].unique():
        mask = df['dow'] == dow
        df.loc[mask, 'dow_avg'] = df.loc[mask, 'call_num'].shift(1).expanding().mean()

    # レジーム変化特徴量
    tax_date = pd.Timestamp('2019-10-01')
    rush_deadline = pd.Timestamp('2019-09-30')

    df['days_to_2019_10_01'] = (tax_date - df['cdr_date']).dt.days
    df['is_post_2019_10_01'] = (df['cdr_date'] >= tax_date).astype(int)
    df['is_post_2019_09_30'] = (df['cdr_date'] >= rush_deadline).astype(int)

    rush_start = rush_deadline - pd.Timedelta(days=90)
    df['is_rush_period'] = ((df['cdr_date'] >= rush_start) &
                            (df['cdr_date'] <= rush_deadline)).astype(int)

    adaptation_end = tax_date + pd.Timedelta(days=30)
    df['is_adaptation_period'] = ((df['cdr_date'] >= tax_date) &
                                   (df['cdr_date'] <= adaptation_end)).astype(int)

    # 翌日の入電数を目的変数
    df['target_next_day'] = df['call_num'].shift(-1)

    print(f"特徴量作成完了: {len(df.columns)}列")

    return df


def get_feature_columns():
    """exp22で使用する特徴量リスト"""
    return [
        'dow', 'day_of_month', 'year',
        'day_of_year', 'week_of_year',
        'is_month_start', 'is_month_end',
        'day_before_holiday_flag',
        'cm_flg', 'acc_get_cnt', 'search_cnt',
        'cm_7d', 'gt_ma_7', 'acc_ma_7', 'dow_avg',
        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_14', 'lag_30',
        'ma_3', 'ma_7', 'ma_30',
        'ma_std_3', 'ma_std_7', 'ma_std_30',
        'days_to_2019_10_01', 'is_post_2019_10_01',
        'is_post_2019_09_30',
        'is_rush_period', 'is_adaptation_period',
    ]


# =============================================================================
# ベースラインモデル
# =============================================================================
def create_baseline_predictions(df):
    """ベースライン予測（lag7, ma7）"""
    df = df.copy()

    # lag7: 前週同曜日の値（0または欠損の場合は代替ラグを使用）
    df['pred_lag7'] = df['lag_7'].copy()

    # lag7が0または欠損の場合、lag14を使用、それでも0ならlag30
    mask_zero_7 = (df['pred_lag7'] == 0) | (df['pred_lag7'].isna())
    df.loc[mask_zero_7, 'pred_lag7'] = df.loc[mask_zero_7, 'lag_14']

    mask_zero_14 = (df['pred_lag7'] == 0) | (df['pred_lag7'].isna())
    df.loc[mask_zero_14, 'pred_lag7'] = df.loc[mask_zero_14, 'lag_30']

    # それでも0の場合はma_7を使用
    mask_still_zero = (df['pred_lag7'] == 0) | (df['pred_lag7'].isna())
    df.loc[mask_still_zero, 'pred_lag7'] = df.loc[mask_still_zero, 'ma_7']

    # ma7: 直近7日平均
    df['pred_ma7'] = df['ma_7']

    return df


# =============================================================================
# CatBoostモデル
# =============================================================================
def train_catboost_model(X_train, y_train, X_test):
    """CatBoostモデルの学習と予測"""
    try:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        return pred, model
    except ImportError:
        print("CatBoostが利用できません。HistGradientBoostingにフォールバック")
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.02,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        return pred, model


# =============================================================================
# 評価関数
# =============================================================================
def calculate_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def evaluate_forecast(y_true, y_pred):
    """予測精度評価"""
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'WAPE': float(calculate_wape(y_true, y_pred))
    }


def evaluate_alert(y_true_alert, y_pred_alert):
    """アラート精度評価"""
    cm = confusion_matrix(y_true_alert, y_pred_alert, labels=[0, 1])

    return {
        'precision': float(precision_score(y_true_alert, y_pred_alert, zero_division=0)),
        'recall': float(recall_score(y_true_alert, y_pred_alert, zero_division=0)),
        'f1': float(f1_score(y_true_alert, y_pred_alert, zero_division=0)),
        'confusion_matrix': cm.tolist()
    }


# =============================================================================
# 高負荷日アラート
# =============================================================================
def calculate_alert_thresholds(train_call_num, test_call_num=None, use_dynamic=False):
    """高負荷日の閾値を計算

    Args:
        train_call_num: 学習期間のコール数
        test_call_num: 検証期間のコール数（動的閾値用）
        use_dynamic: Trueの場合、検証期間内の分布で閾値を決定
    """
    # 学習期間からの閾値（本来の方法）
    train_threshold_a = np.percentile(train_call_num, HIGH_LOAD_PERCENTILE)
    train_mean = train_call_num.mean()
    train_std = train_call_num.std()
    train_threshold_b = train_mean + train_std

    if use_dynamic and test_call_num is not None:
        # 動的閾値：検証期間の分布から計算（評価用、実運用ではリークに注意）
        threshold_a = np.percentile(test_call_num, HIGH_LOAD_PERCENTILE)
        test_mean = test_call_num.mean()
        test_std = test_call_num.std()
        threshold_b = test_mean + test_std

        return {
            'A_top20pct': float(threshold_a),
            'B_mean_plus_1std': float(threshold_b),
            'train_mean': float(train_mean),
            'train_std': float(train_std),
            'train_threshold_A': float(train_threshold_a),
            'train_threshold_B': float(train_threshold_b),
            'test_mean': float(test_mean),
            'test_std': float(test_std),
            'dynamic_threshold': True
        }
    else:
        return {
            'A_top20pct': float(train_threshold_a),
            'B_mean_plus_1std': float(train_threshold_b),
            'train_mean': float(train_mean),
            'train_std': float(train_std),
            'dynamic_threshold': False
        }


def generate_alerts(pred_values, threshold):
    """予測値が閾値を超えたらアラート"""
    return (pred_values > threshold).astype(int)


def get_actual_high_load(actual_values, threshold):
    """実績が閾値を超えたら高負荷日"""
    return (actual_values > threshold).astype(int)


# =============================================================================
# 運用シミュレーション
# =============================================================================
def calculate_staffing(pred_call_num, cap=CAP, buffer=BUFFER):
    """予測値から必要要員を算出"""
    return np.ceil(pred_call_num / cap * buffer).astype(int)


def calculate_daily_metrics(actual, pred, cap=CAP, buffer=BUFFER, c_short=C_SHORT, c_over=C_OVER):
    """日次の不足・過剰・コストを計算"""
    staff_required = calculate_staffing(pred, cap, buffer)
    capacity_calls = staff_required * cap

    short_calls = np.maximum(0, actual - capacity_calls)
    over_calls = np.maximum(0, capacity_calls - actual)
    daily_cost = c_short * short_calls + c_over * over_calls

    return {
        'staff_required': staff_required,
        'capacity_calls': capacity_calls,
        'short_calls': short_calls,
        'over_calls': over_calls,
        'daily_cost': daily_cost
    }


def aggregate_simulation_results(daily_metrics):
    """シミュレーション結果を集計"""
    return {
        'total_cost': float(daily_metrics['daily_cost'].sum()),
        'total_short_calls': float(daily_metrics['short_calls'].sum()),
        'total_over_calls': float(daily_metrics['over_calls'].sum()),
        'staff_avg': float(daily_metrics['staff_required'].mean()),
        'staff_std': float(daily_metrics['staff_required'].std())
    }


# =============================================================================
# 可視化
# =============================================================================
def plot_actual_vs_predicted(df_test, output_dir):
    """実績 vs 予測グラフ"""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = df_test['cdr_date']

    ax.plot(dates, df_test['actual'], 'ko-', label='Actual', markersize=5, linewidth=1.5)
    ax.plot(dates, df_test['pred_catboost'], 'b^-', label='CatBoost', markersize=4, alpha=0.8)
    ax.plot(dates, df_test['pred_lag7'], 'gs--', label='lag7', markersize=3, alpha=0.7)
    ax.plot(dates, df_test['pred_ma7'], 'r+--', label='ma7', markersize=4, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Call Number')
    ax.set_title('Actual vs Predicted (Validation Period)')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"  保存: {filepath}")


def plot_alert_threshold(df_test, threshold_a, threshold_b, output_dir):
    """アラート閾値付きグラフ"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    dates = df_test['cdr_date']
    actual = df_test['actual']
    pred = df_test['pred_catboost']

    # --- 閾値A: 上位20% ---
    ax = axes[0]
    ax.bar(dates, actual, color='lightgray', label='Actual', alpha=0.7, width=0.8)
    ax.plot(dates, pred, 'b-', label='CatBoost Pred', linewidth=2)
    ax.axhline(y=threshold_a, color='red', linestyle='--', linewidth=2, label=f'Threshold A (top20%): {threshold_a:.0f}')

    # 正解/誤報をマーク
    for i, row in df_test.iterrows():
        date = row['cdr_date']
        act = row['actual']
        alert = row['alert_catboost_A']
        true_high = row['true_high_load_A']

        if alert == 1 and true_high == 1:
            ax.scatter(date, act, color='green', s=100, marker='o', zorder=5)  # True Positive
        elif alert == 1 and true_high == 0:
            ax.scatter(date, act, color='orange', s=100, marker='x', zorder=5)  # False Positive
        elif alert == 0 and true_high == 1:
            ax.scatter(date, act, color='red', s=100, marker='s', zorder=5)  # False Negative

    ax.set_ylabel('Call Number')
    ax.set_title('Alert Analysis - Threshold A (Top 20%)')
    ax.legend(loc='upper right')

    # --- 閾値B: 平均+1σ ---
    ax = axes[1]
    ax.bar(dates, actual, color='lightgray', label='Actual', alpha=0.7, width=0.8)
    ax.plot(dates, pred, 'b-', label='CatBoost Pred', linewidth=2)
    ax.axhline(y=threshold_b, color='purple', linestyle='--', linewidth=2, label=f'Threshold B (mean+1std): {threshold_b:.0f}')

    for i, row in df_test.iterrows():
        date = row['cdr_date']
        act = row['actual']
        alert = row['alert_catboost_B']
        true_high = row['true_high_load_B']

        if alert == 1 and true_high == 1:
            ax.scatter(date, act, color='green', s=100, marker='o', zorder=5)
        elif alert == 1 and true_high == 0:
            ax.scatter(date, act, color='orange', s=100, marker='x', zorder=5)
        elif alert == 0 and true_high == 1:
            ax.scatter(date, act, color='red', s=100, marker='s', zorder=5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Call Number')
    ax.set_title('Alert Analysis - Threshold B (Mean + 1 Std)')
    ax.legend(loc='upper right')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)

    # 凡例追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Positive (Hit)'),
        Line2D([0], [0], marker='x', color='orange', markersize=10, label='False Positive (False Alarm)', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='False Negative (Miss)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    filepath = os.path.join(output_dir, 'alert_threshold.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"  保存: {filepath}")


def plot_cumulative_cost(df_test, output_dir):
    """累積コストグラフ"""
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = df_test['cdr_date']

    cum_cost_catboost = df_test['cost_catboost'].cumsum()
    cum_cost_lag7 = df_test['cost_lag7'].cumsum()
    cum_cost_ma7 = df_test['cost_ma7'].cumsum()

    ax.plot(dates, cum_cost_catboost, 'b-', label='CatBoost', linewidth=2)
    ax.plot(dates, cum_cost_lag7, 'g--', label='lag7', linewidth=2)
    ax.plot(dates, cum_cost_ma7, 'r:', label='ma7', linewidth=2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Cost (JPY)')
    ax.set_title('Cumulative Staffing Cost by Model')
    ax.legend(loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)

    # 最終コストを表示
    final_catboost = cum_cost_catboost.iloc[-1]
    final_lag7 = cum_cost_lag7.iloc[-1]
    final_ma7 = cum_cost_ma7.iloc[-1]

    ax.annotate(f'{final_catboost:,.0f}', xy=(dates.iloc[-1], final_catboost),
                xytext=(5, 0), textcoords='offset points', fontsize=10, color='blue')
    ax.annotate(f'{final_lag7:,.0f}', xy=(dates.iloc[-1], final_lag7),
                xytext=(5, 0), textcoords='offset points', fontsize=10, color='green')
    ax.annotate(f'{final_ma7:,.0f}', xy=(dates.iloc[-1], final_ma7),
                xytext=(5, 0), textcoords='offset points', fontsize=10, color='red')

    plt.tight_layout()

    filepath = os.path.join(output_dir, 'cumulative_cost.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"  保存: {filepath}")


# =============================================================================
# 出力生成
# =============================================================================
def save_json(data, filepath):
    """JSONファイル保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  保存: {filepath}")


def generate_slide_summary(
    forecast_metrics,
    alert_metrics,
    sim_summary,
    thresholds,
    validation_period,
    output_dir
):
    """スライド用サマリーMarkdown生成"""

    best_model = min(sim_summary, key=lambda x: sim_summary[x]['total_cost'])
    best_cost = sim_summary[best_model]['total_cost']
    worst_cost = max(sim_summary[m]['total_cost'] for m in sim_summary)
    cost_reduction = (1 - best_cost / worst_cost) * 100 if worst_cost > 0 else 0

    # 最も予測精度の良いモデルを決定
    best_forecast_model = min(forecast_metrics, key=lambda x: forecast_metrics[x]['MAE'])

    md_content = f"""# コールセンター入電数予測 - 高負荷日アラート評価レポート

## 検証概要
- **検証期間**: {validation_period['start']} ～ {validation_period['end']} ({validation_period['days']}日間)
- **対象**: 平日のみ（土日祝除く）
- **目的変数**: 翌営業日の入電数（call_num）

---

## 1. 予測精度（MAE / WAPE）

| モデル | MAE | WAPE |
|--------|-----|------|
| **CatBoost** | {forecast_metrics['catboost']['MAE']:.1f} | {forecast_metrics['catboost']['WAPE']:.1f}% |
| lag7 | {forecast_metrics['lag7']['MAE']:.1f} | {forecast_metrics['lag7']['WAPE']:.1f}% |
| ma7 | {forecast_metrics['ma7']['MAE']:.1f} | {forecast_metrics['ma7']['WAPE']:.1f}% |

**結論**: 検証期間では{best_forecast_model}が最も予測誤差が小さい（ただし期間によって変動）。日次の入電数計画に活用可能。

---

## 2. 高負荷日アラート精度

### 閾値定義
- **A) 上位20%**: {thresholds['A_top20pct']:.0f} コール以上
- **B) 平均+1σ**: {thresholds['B_mean_plus_1std']:.0f} コール以上
  - (学習期間: 平均={thresholds['train_mean']:.1f}, σ={thresholds['train_std']:.1f})
  - {'**注意**: 動的閾値モード使用（検証期間の分布から閾値を決定）' if thresholds.get('dynamic_threshold') else ''}

### 閾値A（上位20%）での結果

| モデル | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| CatBoost | {alert_metrics['catboost_A']['precision']:.2f} | {alert_metrics['catboost_A']['recall']:.2f} | {alert_metrics['catboost_A']['f1']:.2f} |
| lag7 | {alert_metrics['lag7_A']['precision']:.2f} | {alert_metrics['lag7_A']['recall']:.2f} | {alert_metrics['lag7_A']['f1']:.2f} |
| ma7 | {alert_metrics['ma7_A']['precision']:.2f} | {alert_metrics['ma7_A']['recall']:.2f} | {alert_metrics['ma7_A']['f1']:.2f} |

### 閾値B（平均+1σ）での結果

| モデル | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| CatBoost | {alert_metrics['catboost_B']['precision']:.2f} | {alert_metrics['catboost_B']['recall']:.2f} | {alert_metrics['catboost_B']['f1']:.2f} |
| lag7 | {alert_metrics['lag7_B']['precision']:.2f} | {alert_metrics['lag7_B']['recall']:.2f} | {alert_metrics['lag7_B']['f1']:.2f} |
| ma7 | {alert_metrics['ma7_B']['precision']:.2f} | {alert_metrics['ma7_B']['recall']:.2f} | {alert_metrics['ma7_B']['f1']:.2f} |

**結論**:
- アラートは予測の"補助（優先度付け/実行漏れ防止）"として活用
- 日次の要員計画は予測値そのものを使用

---

## 3. 運用シミュレーション

### 前提パラメータ
- 1人あたり対応可能コール数: **{CAP}コール/日**
- 安全余裕係数: **{BUFFER}**
- 不足1コールのコスト: **{C_SHORT}円**（外注/残業/機会損失の代理）
- 過剰1コールのコスト: **{C_OVER}円**（ムダ配置の代理）

### 結果サマリー

| モデル | 総コスト | 不足コール計 | 過剰コール計 | 要員平均 | 要員標準偏差 |
|--------|----------|--------------|--------------|----------|--------------|
| CatBoost | {sim_summary['catboost']['total_cost']:,.0f}円 | {sim_summary['catboost']['total_short_calls']:.0f} | {sim_summary['catboost']['total_over_calls']:.0f} | {sim_summary['catboost']['staff_avg']:.1f}人 | {sim_summary['catboost']['staff_std']:.1f} |
| lag7 | {sim_summary['lag7']['total_cost']:,.0f}円 | {sim_summary['lag7']['total_short_calls']:.0f} | {sim_summary['lag7']['total_over_calls']:.0f} | {sim_summary['lag7']['staff_avg']:.1f}人 | {sim_summary['lag7']['staff_std']:.1f} |
| ma7 | {sim_summary['ma7']['total_cost']:,.0f}円 | {sim_summary['ma7']['total_short_calls']:.0f} | {sim_summary['ma7']['total_over_calls']:.0f} | {sim_summary['ma7']['staff_avg']:.1f}人 | {sim_summary['ma7']['staff_std']:.1f} |

**結論**:
- **{best_model}** が最もコスト効率が良い
- 最悪モデル比で約 **{cost_reduction:.1f}%** のコスト削減

---

## 4. 注意事項

1. **アラートの位置づけ**
   - アラートは予測の"補助"であり、日次の要員計画は予測値そのものを使用
   - 高負荷日アラートは「優先度付け」「実行漏れ防止」の用途

2. **パラメータの感度**
   - c_short={C_SHORT}, c_over={C_OVER}, cap={CAP}, buffer={BUFFER} は代理パラメータ
   - 実運用では感度分析を推奨

3. **閾値の選択**
   - 閾値A（上位20%）: より厳選されたアラート
   - 閾値B（平均+1σ）: より広めのアラート
   - 運用目的に応じて選択

4. **閾値設定について**
   - 本評価では検証期間内の分布から閾値を動的に設定（評価目的）
   - 実運用では学習期間の閾値を使用するか、直近N日の分布から閾値を更新する方式を推奨
   - 検証期間（2020年1-3月）はCOVID-19影響で入電数が全体的に低下していた可能性あり

---

## 5. 出力ファイル一覧

- `metrics_forecast.json`: 予測精度指標
- `metrics_alert.json`: アラート精度指標
- `simulation_summary.csv`: シミュレーション結果サマリー
- `simulation_daily_detail.csv`: 日次詳細データ
- `actual_vs_predicted.png`: 実績 vs 予測グラフ
- `alert_threshold.png`: アラート閾値と当たり外れ
- `cumulative_cost.png`: 累積コスト推移

---

*Generated by run_alert_eval.py*
"""

    filepath = os.path.join(output_dir, 'slide_ready_summary.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  保存: {filepath}")


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print("=" * 80)
    print("コールセンター入電数予測 - 高負荷日アラート評価")
    print("=" * 80)

    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n出力ディレクトリ: {OUTPUT_DIR}")

    # ----- Step 1: データ読み込み -----
    df = load_and_preprocess_data(INPUT_DIR)

    # ----- Step 2: 特徴量作成 -----
    df = create_features(df)

    # 平日のみ抽出
    df_model = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)
    print(f"\n平日データ数: {len(df_model)}行")
    print(f"期間: {df_model['cdr_date'].min()} ~ {df_model['cdr_date'].max()}")

    # 特徴量リスト
    feature_cols = get_feature_columns()

    # 欠損除去
    df_clean = df_model.dropna(subset=feature_cols + ['target_next_day']).copy().reset_index(drop=True)
    print(f"欠損除去後: {len(df_clean)}行")

    # ----- Step 3: Holdout分割 -----
    print("\n" + "=" * 80)
    print("Step 3: Holdout分割")
    print("=" * 80)

    # 末尾N日を検証期間とする
    n_test = min(VALIDATION_DAYS, len(df_clean) // 5)  # 最大でも20%まで
    test_start_idx = len(df_clean) - n_test

    train_df = df_clean.iloc[:test_start_idx].copy()
    test_df = df_clean.iloc[test_start_idx:].copy()

    print(f"Train: {len(train_df)}件 ({train_df['cdr_date'].min().strftime('%Y-%m-%d')} ~ {train_df['cdr_date'].max().strftime('%Y-%m-%d')})")
    print(f"Test:  {len(test_df)}件 ({test_df['cdr_date'].min().strftime('%Y-%m-%d')} ~ {test_df['cdr_date'].max().strftime('%Y-%m-%d')})")

    validation_period = {
        'start': test_df['cdr_date'].min().strftime('%Y-%m-%d'),
        'end': test_df['cdr_date'].max().strftime('%Y-%m-%d'),
        'days': len(test_df)
    }

    X_train = train_df[feature_cols]
    y_train = train_df['target_next_day']
    X_test = test_df[feature_cols]
    y_test = test_df['target_next_day']

    # ----- Step 4: モデル学習・予測 -----
    print("\n" + "=" * 80)
    print("Step 4: モデル学習・予測")
    print("=" * 80)

    # ベースライン
    test_df = create_baseline_predictions(test_df)

    # CatBoost
    print("\nCatBoostモデル学習中...")
    pred_catboost, catboost_model = train_catboost_model(X_train, y_train, X_test)
    test_df['pred_catboost'] = pred_catboost

    # 実績値カラム追加
    test_df['actual'] = y_test.values

    # ----- Step 5: 予測精度評価 -----
    print("\n" + "=" * 80)
    print("Step 5: 予測精度評価")
    print("=" * 80)

    forecast_metrics = {
        'catboost': evaluate_forecast(test_df['actual'], test_df['pred_catboost']),
        'lag7': evaluate_forecast(test_df['actual'].dropna(), test_df.loc[test_df['actual'].notna(), 'pred_lag7'].dropna()),
        'ma7': evaluate_forecast(test_df['actual'].dropna(), test_df.loc[test_df['actual'].notna(), 'pred_ma7'].dropna())
    }

    # lag7, ma7の欠損対応
    test_df['pred_lag7'] = test_df['pred_lag7'].fillna(test_df['pred_ma7'])
    test_df['pred_ma7'] = test_df['pred_ma7'].fillna(test_df['pred_lag7'])
    test_df = test_df.dropna(subset=['pred_lag7', 'pred_ma7', 'pred_catboost', 'actual']).reset_index(drop=True)

    # 再計算
    forecast_metrics = {
        'catboost': evaluate_forecast(test_df['actual'], test_df['pred_catboost']),
        'lag7': evaluate_forecast(test_df['actual'], test_df['pred_lag7']),
        'ma7': evaluate_forecast(test_df['actual'], test_df['pred_ma7'])
    }

    # 祝日・休日（call_num=0）を除外したデータフレームも作成
    test_df_workdays = test_df[test_df['actual'] > 0].copy().reset_index(drop=True)
    print(f"\n  (参考) 営業日のみ: {len(test_df_workdays)}日")

    # 営業日のみでの予測精度
    forecast_metrics_workdays = {
        'catboost': evaluate_forecast(test_df_workdays['actual'], test_df_workdays['pred_catboost']),
        'lag7': evaluate_forecast(test_df_workdays['actual'], test_df_workdays['pred_lag7']),
        'ma7': evaluate_forecast(test_df_workdays['actual'], test_df_workdays['pred_ma7'])
    }

    print("\n予測精度（営業日のみ）:")
    print(f"  CatBoost: MAE={forecast_metrics_workdays['catboost']['MAE']:.2f}, WAPE={forecast_metrics_workdays['catboost']['WAPE']:.2f}%")
    print(f"  lag7:     MAE={forecast_metrics_workdays['lag7']['MAE']:.2f}, WAPE={forecast_metrics_workdays['lag7']['WAPE']:.2f}%")
    print(f"  ma7:      MAE={forecast_metrics_workdays['ma7']['MAE']:.2f}, WAPE={forecast_metrics_workdays['ma7']['WAPE']:.2f}%")

    print("\n予測精度:")
    print(f"  CatBoost: MAE={forecast_metrics['catboost']['MAE']:.2f}, WAPE={forecast_metrics['catboost']['WAPE']:.2f}%")
    print(f"  lag7:     MAE={forecast_metrics['lag7']['MAE']:.2f}, WAPE={forecast_metrics['lag7']['WAPE']:.2f}%")
    print(f"  ma7:      MAE={forecast_metrics['ma7']['MAE']:.2f}, WAPE={forecast_metrics['ma7']['WAPE']:.2f}%")

    # ----- Step 6: 高負荷日アラート -----
    print("\n" + "=" * 80)
    print("Step 6: 高負荷日アラート評価")
    print("=" * 80)

    # アラート評価は営業日のみで実施
    test_df_alert = test_df[test_df['actual'] > 0].copy().reset_index(drop=True)

    # 閾値計算（営業日のみ）
    train_workdays = train_df[train_df['call_num'] > 0]['call_num']
    thresholds = calculate_alert_thresholds(
        train_workdays,
        test_df_alert['actual'],
        use_dynamic=USE_DYNAMIC_THRESHOLD
    )

    if thresholds.get('dynamic_threshold'):
        print(f"\n【動的閾値モード】検証期間の分布から閾値を決定（評価用）")
        print(f"閾値A (上位{100-HIGH_LOAD_PERCENTILE}%): {thresholds['A_top20pct']:.0f}")
        print(f"閾値B (平均+1σ): {thresholds['B_mean_plus_1std']:.0f}")
        print(f"  検証期間: 平均={thresholds['test_mean']:.1f}, σ={thresholds['test_std']:.1f}")
        print(f"  (参考) 学習期間閾値A: {thresholds['train_threshold_A']:.0f}, B: {thresholds['train_threshold_B']:.0f}")
    else:
        print(f"\n閾値A (上位{100-HIGH_LOAD_PERCENTILE}%): {thresholds['A_top20pct']:.0f}")
        print(f"閾値B (平均+1σ): {thresholds['B_mean_plus_1std']:.0f}")
        print(f"  学習期間平均: {thresholds['train_mean']:.1f}, σ: {thresholds['train_std']:.1f}")

    # 閾値A
    threshold_a = thresholds['A_top20pct']
    test_df_alert['true_high_load_A'] = get_actual_high_load(test_df_alert['actual'], threshold_a)
    test_df_alert['alert_catboost_A'] = generate_alerts(test_df_alert['pred_catboost'], threshold_a)
    test_df_alert['alert_lag7_A'] = generate_alerts(test_df_alert['pred_lag7'], threshold_a)
    test_df_alert['alert_ma7_A'] = generate_alerts(test_df_alert['pred_ma7'], threshold_a)

    # 閾値B
    threshold_b = thresholds['B_mean_plus_1std']
    test_df_alert['true_high_load_B'] = get_actual_high_load(test_df_alert['actual'], threshold_b)
    test_df_alert['alert_catboost_B'] = generate_alerts(test_df_alert['pred_catboost'], threshold_b)
    test_df_alert['alert_lag7_B'] = generate_alerts(test_df_alert['pred_lag7'], threshold_b)
    test_df_alert['alert_ma7_B'] = generate_alerts(test_df_alert['pred_ma7'], threshold_b)

    # アラート精度評価（営業日のみ）
    alert_metrics = {
        'catboost_A': evaluate_alert(test_df_alert['true_high_load_A'], test_df_alert['alert_catboost_A']),
        'lag7_A': evaluate_alert(test_df_alert['true_high_load_A'], test_df_alert['alert_lag7_A']),
        'ma7_A': evaluate_alert(test_df_alert['true_high_load_A'], test_df_alert['alert_ma7_A']),
        'catboost_B': evaluate_alert(test_df_alert['true_high_load_B'], test_df_alert['alert_catboost_B']),
        'lag7_B': evaluate_alert(test_df_alert['true_high_load_B'], test_df_alert['alert_lag7_B']),
        'ma7_B': evaluate_alert(test_df_alert['true_high_load_B'], test_df_alert['alert_ma7_B']),
    }

    print(f"\n営業日のみで評価: {len(test_df_alert)}日")

    print("\nアラート精度（閾値A: 上位20%）:")
    for model in ['catboost', 'lag7', 'ma7']:
        m = alert_metrics[f'{model}_A']
        print(f"  {model}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

    print("\nアラート精度（閾値B: 平均+1σ）:")
    for model in ['catboost', 'lag7', 'ma7']:
        m = alert_metrics[f'{model}_B']
        print(f"  {model}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")

    # ----- Step 7: 運用シミュレーション -----
    print("\n" + "=" * 80)
    print("Step 7: 運用シミュレーション")
    print("=" * 80)

    sim_summary = {}

    for model_name, pred_col in [('catboost', 'pred_catboost'), ('lag7', 'pred_lag7'), ('ma7', 'pred_ma7')]:
        daily_metrics = calculate_daily_metrics(test_df['actual'].values, test_df[pred_col].values)

        test_df[f'staff_{model_name}'] = daily_metrics['staff_required']
        test_df[f'short_{model_name}'] = daily_metrics['short_calls']
        test_df[f'over_{model_name}'] = daily_metrics['over_calls']
        test_df[f'cost_{model_name}'] = daily_metrics['daily_cost']

        sim_summary[model_name] = aggregate_simulation_results(pd.DataFrame(daily_metrics))

    print("\nシミュレーション結果:")
    print(f"{'Model':<12} {'Total Cost':>12} {'Short':>8} {'Over':>8} {'Staff Avg':>10} {'Staff Std':>10}")
    print("-" * 62)
    for model_name, results in sim_summary.items():
        print(f"{model_name:<12} {results['total_cost']:>12,.0f} {results['total_short_calls']:>8,.0f} {results['total_over_calls']:>8,.0f} {results['staff_avg']:>10.1f} {results['staff_std']:>10.1f}")

    # ----- Step 8: 出力生成 -----
    print("\n" + "=" * 80)
    print("Step 8: 出力ファイル生成")
    print("=" * 80)

    # test_dfにアラート情報を追加（全日用）
    test_df['true_high_load_A'] = get_actual_high_load(test_df['actual'], threshold_a)
    test_df['alert_catboost_A'] = generate_alerts(test_df['pred_catboost'], threshold_a)
    test_df['true_high_load_B'] = get_actual_high_load(test_df['actual'], threshold_b)
    test_df['alert_catboost_B'] = generate_alerts(test_df['pred_catboost'], threshold_b)
    test_df['alert_lag7_A'] = generate_alerts(test_df['pred_lag7'], threshold_a)
    test_df['alert_ma7_A'] = generate_alerts(test_df['pred_ma7'], threshold_a)
    test_df['alert_lag7_B'] = generate_alerts(test_df['pred_lag7'], threshold_b)
    test_df['alert_ma7_B'] = generate_alerts(test_df['pred_ma7'], threshold_b)

    # JSON
    save_json(forecast_metrics, os.path.join(OUTPUT_DIR, 'metrics_forecast.json'))
    save_json({**alert_metrics, 'thresholds': thresholds}, os.path.join(OUTPUT_DIR, 'metrics_alert.json'))

    # CSV - シミュレーションサマリー
    sim_summary_df = pd.DataFrame(sim_summary).T
    sim_summary_df.index.name = 'model'
    sim_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'simulation_summary.csv'))
    print(f"  保存: {os.path.join(OUTPUT_DIR, 'simulation_summary.csv')}")

    # CSV - 日次詳細
    daily_detail_cols = [
        'cdr_date', 'actual',
        'pred_catboost', 'pred_lag7', 'pred_ma7',
        'true_high_load_A', 'true_high_load_B',
        'alert_catboost_A', 'alert_lag7_A', 'alert_ma7_A',
        'alert_catboost_B', 'alert_lag7_B', 'alert_ma7_B',
        'staff_catboost', 'staff_lag7', 'staff_ma7',
        'short_catboost', 'short_lag7', 'short_ma7',
        'over_catboost', 'over_lag7', 'over_ma7',
        'cost_catboost', 'cost_lag7', 'cost_ma7'
    ]
    test_df[daily_detail_cols].to_csv(os.path.join(OUTPUT_DIR, 'simulation_daily_detail.csv'), index=False)
    print(f"  保存: {os.path.join(OUTPUT_DIR, 'simulation_daily_detail.csv')}")

    # PNG
    print("\nグラフ生成中...")
    # 営業日のみのデータで可視化
    test_df_viz = test_df[test_df['actual'] > 0].copy().reset_index(drop=True)
    plot_actual_vs_predicted(test_df_viz, OUTPUT_DIR)
    plot_alert_threshold(test_df_viz, threshold_a, threshold_b, OUTPUT_DIR)
    plot_cumulative_cost(test_df_viz, OUTPUT_DIR)

    # 営業日のみのシミュレーション結果
    sim_summary_workdays = {}
    test_df_work = test_df[test_df['actual'] > 0].copy()
    for model_name, pred_col in [('catboost', 'pred_catboost'), ('lag7', 'pred_lag7'), ('ma7', 'pred_ma7')]:
        daily_metrics = calculate_daily_metrics(test_df_work['actual'].values, test_df_work[pred_col].values)
        sim_summary_workdays[model_name] = aggregate_simulation_results(pd.DataFrame(daily_metrics))

    # Markdown
    generate_slide_summary(
        forecast_metrics_workdays,  # 営業日のみの予測精度を使用
        alert_metrics,
        sim_summary,
        thresholds,
        validation_period,
        OUTPUT_DIR
    )

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)
    print(f"\n出力先: {OUTPUT_DIR}/")
    print("  - metrics_forecast.json")
    print("  - metrics_alert.json")
    print("  - simulation_summary.csv")
    print("  - simulation_daily_detail.csv")
    print("  - slide_ready_summary.md")
    print("  - actual_vs_predicted.png")
    print("  - alert_threshold.png")
    print("  - cumulative_cost.png")


if __name__ == '__main__':
    main()

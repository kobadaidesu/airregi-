"""
Validation vs Test スコア差の原因調査スクリプト

調査内容:
1. データ分割の期間と件数
2. Train/Val/Test の目的変数の分布
3. 時系列の特徴（トレンド、季節性）
4. データリーケージの可能性
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
print("=" * 80)
print("データ読み込み")
print("=" * 80)

calender = pd.read_csv('../input/calender_data.csv')
cm_data = pd.read_csv('../input/cm_data.csv')
gt_service = pd.read_csv('../input/gt_service_name.csv')
acc_get = pd.read_csv('../input/regi_acc_get_data_transform.csv')
call_data = pd.read_csv('../input/regi_call_data_transform.csv')

# 日付型に変換
calender['cdr_date'] = pd.to_datetime(calender['cdr_date'])
cm_data['cdr_date'] = pd.to_datetime(cm_data['cdr_date'])
acc_get['cdr_date'] = pd.to_datetime(acc_get['cdr_date'])
call_data['cdr_date'] = pd.to_datetime(call_data['cdr_date'])
gt_service['week'] = pd.to_datetime(gt_service['week'])

# データマージ
df = call_data.copy()
df = df.merge(calender, on='cdr_date', how='left')
df = df.merge(cm_data, on='cdr_date', how='left')
df = df.merge(acc_get, on='cdr_date', how='left')

# Google Trendsを日次に展開
gt_service_daily = []
for idx, row in gt_service.iterrows():
    week_start = row['week']
    for i in range(7):
        date = week_start + timedelta(days=i)
        gt_service_daily.append({
            'cdr_date': date,
            'search_cnt': row['search_cnt']
        })

gt_daily = pd.DataFrame(gt_service_daily)
df = df.merge(gt_daily, on='cdr_date', how='left')

# 翌日の入電数を目的変数にする
df['target_next_day'] = df['call_num'].shift(-1)
df = df.dropna(subset=['target_next_day']).reset_index(drop=True)

# 平日のみ
df_model = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

print(f"\n平日データ数: {len(df_model)}行")
print(f"期間: {df_model['cdr_date'].min()} ~ {df_model['cdr_date'].max()}")

# ==================================================================================
# 調査1: データ分割の詳細
# ==================================================================================
print("\n" + "=" * 80)
print("調査1: データ分割の詳細")
print("=" * 80)

val_months = 2
test_months = 2

max_date = df_model['cdr_date'].max()
test_start = max_date - pd.Timedelta(days=30*test_months)
val_start = test_start - pd.Timedelta(days=30*val_months)

train_df = df_model[df_model['cdr_date'] < val_start].copy()
val_df = df_model[(df_model['cdr_date'] >= val_start) & (df_model['cdr_date'] < test_start)].copy()
test_df = df_model[df_model['cdr_date'] >= test_start].copy()

print(f"\nTrain:")
print(f"  期間: {train_df['cdr_date'].min()} ~ {train_df['cdr_date'].max()}")
print(f"  件数: {len(train_df)}行")
print(f"  日数: {(train_df['cdr_date'].max() - train_df['cdr_date'].min()).days}日")

print(f"\nValidation:")
print(f"  期間: {val_df['cdr_date'].min()} ~ {val_df['cdr_date'].max()}")
print(f"  件数: {len(val_df)}行")
print(f"  日数: {(val_df['cdr_date'].max() - val_df['cdr_date'].min()).days}日")

print(f"\nTest:")
print(f"  期間: {test_df['cdr_date'].min()} ~ {test_df['cdr_date'].max()}")
print(f"  件数: {len(test_df)}行")
print(f"  日数: {(test_df['cdr_date'].max() - test_df['cdr_date'].min()).days}日")

# ==================================================================================
# 調査2: 目的変数の分布比較
# ==================================================================================
print("\n" + "=" * 80)
print("調査2: 目的変数（翌日入電数）の分布比較")
print("=" * 80)

print("\nTrain統計:")
print(train_df['target_next_day'].describe())

print("\nValidation統計:")
print(val_df['target_next_day'].describe())

print("\nTest統計:")
print(test_df['target_next_day'].describe())

# 分布の可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ヒストグラム
ax = axes[0, 0]
ax.hist(train_df['target_next_day'], bins=30, alpha=0.5, label='Train', color='blue')
ax.hist(val_df['target_next_day'], bins=30, alpha=0.5, label='Validation', color='orange')
ax.hist(test_df['target_next_day'], bins=30, alpha=0.5, label='Test', color='green')
ax.set_xlabel('Target (Next Day Calls)')
ax.set_ylabel('Frequency')
ax.set_title('Target Distribution Comparison')
ax.legend()
ax.grid(alpha=0.3)

# 2. ボックスプロット
ax = axes[0, 1]
data_to_plot = [
    train_df['target_next_day'],
    val_df['target_next_day'],
    test_df['target_next_day']
]
ax.boxplot(data_to_plot, labels=['Train', 'Validation', 'Test'])
ax.set_ylabel('Target (Next Day Calls)')
ax.set_title('Target Distribution (Box Plot)')
ax.grid(alpha=0.3)

# 3. 時系列プロット
ax = axes[1, 0]
ax.plot(train_df['cdr_date'], train_df['target_next_day'], label='Train', alpha=0.7, linewidth=0.8)
ax.plot(val_df['cdr_date'], val_df['target_next_day'], label='Validation', alpha=0.7, linewidth=0.8)
ax.plot(test_df['cdr_date'], test_df['target_next_day'], label='Test', alpha=0.7, linewidth=0.8)
ax.axvline(val_start, color='orange', linestyle='--', alpha=0.5, label='Val Start')
ax.axvline(test_start, color='green', linestyle='--', alpha=0.5, label='Test Start')
ax.set_xlabel('Date')
ax.set_ylabel('Target (Next Day Calls)')
ax.set_title('Time Series View')
ax.legend()
ax.grid(alpha=0.3)

# 4. 移動平均プロット（トレンド確認）
ax = axes[1, 1]
train_ma = train_df.set_index('cdr_date')['target_next_day'].rolling(window=7).mean()
val_ma = val_df.set_index('cdr_date')['target_next_day'].rolling(window=7).mean()
test_ma = test_df.set_index('cdr_date')['target_next_day'].rolling(window=7).mean()

ax.plot(train_ma.index, train_ma.values, label='Train (7-day MA)', alpha=0.7)
ax.plot(val_ma.index, val_ma.values, label='Validation (7-day MA)', alpha=0.7)
ax.plot(test_ma.index, test_ma.values, label='Test (7-day MA)', alpha=0.7)
ax.axvline(val_start, color='orange', linestyle='--', alpha=0.5)
ax.axvline(test_start, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Target (7-day Moving Average)')
ax.set_title('Trend Comparison (Moving Average)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../output/exp13/score_diff_investigation.png', dpi=300, bbox_inches='tight')
print("\n可視化を保存: ../output/exp13/score_diff_investigation.png")
plt.close()

# ==================================================================================
# 調査3: 統計的検定（分布が同じか？）
# ==================================================================================
print("\n" + "=" * 80)
print("調査3: 分布の類似性（平均値の差）")
print("=" * 80)

train_mean = train_df['target_next_day'].mean()
val_mean = val_df['target_next_day'].mean()
test_mean = test_df['target_next_day'].mean()

print(f"\n平均値:")
print(f"  Train     : {train_mean:.2f}")
print(f"  Validation: {val_mean:.2f}")
print(f"  Test      : {test_mean:.2f}")

print(f"\n平均値の差:")
print(f"  Val - Train : {val_mean - train_mean:.2f} ({(val_mean - train_mean) / train_mean * 100:.1f}%)")
print(f"  Test - Train: {test_mean - train_mean:.2f} ({(test_mean - train_mean) / train_mean * 100:.1f}%)")
print(f"  Test - Val  : {test_mean - val_mean:.2f} ({(test_mean - val_mean) / val_mean * 100:.1f}%)")

train_std = train_df['target_next_day'].std()
val_std = val_df['target_next_day'].std()
test_std = test_df['target_next_day'].std()

print(f"\n標準偏差:")
print(f"  Train     : {train_std:.2f}")
print(f"  Validation: {val_std:.2f}")
print(f"  Test      : {test_std:.2f}")

# ==================================================================================
# 調査4: 特定の期間の特徴
# ==================================================================================
print("\n" + "=" * 80)
print("調査4: 各期間の特徴（月別、曜日別）")
print("=" * 80)

print("\n【Validation期間の特徴】")
print("月別平均:")
print(val_df.groupby('month')['target_next_day'].mean())
print("\n曜日別平均:")
print(val_df.groupby('dow')['target_next_day'].mean())

print("\n【Test期間の特徴】")
print("月別平均:")
print(test_df.groupby('month')['target_next_day'].mean())
print("\n曜日別平均:")
print(test_df.groupby('dow')['target_next_day'].mean())

# ==================================================================================
# 調査5: 外部要因（CM、アカウント取得、Google Trends）
# ==================================================================================
print("\n" + "=" * 80)
print("調査5: 外部要因の比較")
print("=" * 80)

print("\n【CM実施状況】")
print(f"  Train     : {train_df['cm_flg'].sum()}回")
print(f"  Validation: {val_df['cm_flg'].sum()}回")
print(f"  Test      : {test_df['cm_flg'].sum()}回")

print("\n【アカウント取得数の平均】")
print(f"  Train     : {train_df['acc_get_cnt'].mean():.2f}")
print(f"  Validation: {val_df['acc_get_cnt'].mean():.2f}")
print(f"  Test      : {test_df['acc_get_cnt'].mean():.2f}")

print("\n【Google Trends検索数の平均】")
print(f"  Train     : {train_df['search_cnt'].mean():.2f}")
print(f"  Validation: {val_df['search_cnt'].mean():.2f}")
print(f"  Test      : {test_df['search_cnt'].mean():.2f}")

# ==================================================================================
# 調査6: データの詳細期間
# ==================================================================================
print("\n" + "=" * 80)
print("調査6: データの詳細期間（年月で確認）")
print("=" * 80)

print("\nTrain期間:")
print(f"  開始: {train_df['cdr_date'].min().strftime('%Y年%m月%d日')}")
print(f"  終了: {train_df['cdr_date'].max().strftime('%Y年%m月%d日')}")

print("\nValidation期間:")
print(f"  開始: {val_df['cdr_date'].min().strftime('%Y年%m月%d日')}")
print(f"  終了: {val_df['cdr_date'].max().strftime('%Y年%m月%d日')}")
print(f"  ※ この期間は {val_df['cdr_date'].min().strftime('%Y年%m月')}~{val_df['cdr_date'].max().strftime('%Y年%m月')} にあたる")

print("\nTest期間:")
print(f"  開始: {test_df['cdr_date'].min().strftime('%Y年%m月%d日')}")
print(f"  終了: {test_df['cdr_date'].max().strftime('%Y年%m月%d日')}")
print(f"  ※ この期間は {test_df['cdr_date'].min().strftime('%Y年%m月')}~{test_df['cdr_date'].max().strftime('%Y年%m月')} にあたる")

# ==================================================================================
# まとめ
# ==================================================================================
print("\n" + "=" * 80)
print("調査結果まとめ")
print("=" * 80)

print("\n【考えられる原因】")
print("1. データの時系列トレンド:")
print("   - Validation期間とTest期間で入電数の平均値が大きく異なる可能性")
print("   - 季節性やビジネスサイクルの影響")

print("\n2. データ分布の違い:")
print("   - ValidationがTrainと分布が異なり、Testがより似ている可能性")
print("   - または逆に、ValidationがTrainに近く、Testが異なる可能性")

print("\n3. 外部要因の影響:")
print("   - CM実施頻度の違い")
print("   - アカウント取得数のトレンド変化")
print("   - Google Trendsの検索トレンド変化")

print("\n4. サンプル数の違い:")
print(f"   - Validation: {len(val_df)}件")
print(f"   - Test: {len(test_df)}件")
print("   - サンプル数が少ないとスコアが不安定になる")

print("\n【次のアクション】")
print("- 上記の可視化を確認してデータの特性を把握")
print("- 特にValidation期間が特殊な時期（年末年始、決算期など）に該当していないか確認")
print("- モデルがValidation期間の特徴に過学習している可能性も考慮")

print("\n" + "=" * 80)

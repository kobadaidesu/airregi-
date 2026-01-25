"""
exp22 リーク詳細検証スクリプト (check_model2)
==============================================

検証項目:
1. dow_avg - 曜日別平均が全期間で計算されている問題
2. rolling特徴量のshift(1)が適切に機能しているか
3. acc_get_cnt, search_cntなどの外部データが「当日の値」を使っている可能性
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

# ==================================================================================
# データの読み込み
# ==================================================================================

def load_data():
    calender = pd.read_csv('../input/calender_data.csv')
    cm_data = pd.read_csv('../input/cm_data.csv')
    gt_service = pd.read_csv('../input/gt_service_name.csv')
    acc_get = pd.read_csv('../input/regi_acc_get_data_transform.csv')
    call_data = pd.read_csv('../input/regi_call_data_transform.csv')

    calender['cdr_date'] = pd.to_datetime(calender['cdr_date'])
    cm_data['cdr_date'] = pd.to_datetime(cm_data['cdr_date'])
    acc_get['cdr_date'] = pd.to_datetime(acc_get['cdr_date'])
    call_data['cdr_date'] = pd.to_datetime(call_data['cdr_date'])
    gt_service['week'] = pd.to_datetime(gt_service['week'])

    return calender, cm_data, gt_service, acc_get, call_data


def merge_base_data(calender, cm_data, gt_service, acc_get, call_data):
    df = call_data.copy()
    df = df.merge(calender, on='cdr_date', how='left')
    df = df.merge(cm_data, on='cdr_date', how='left')
    df = df.merge(acc_get, on='cdr_date', how='left')

    gt_service_daily = []
    for idx, row in gt_service.iterrows():
        week_start = row['week']
        for i in range(7):
            date = week_start + timedelta(days=i)
            gt_service_daily.append({'cdr_date': date, 'search_cnt': row['search_cnt']})

    gt_daily = pd.DataFrame(gt_service_daily)
    df = df.merge(gt_daily, on='cdr_date', how='left')

    return df


# ExtraTreesのパラメータ
EXTRA_TREES_PARAMS = {
    'n_estimators': 274,
    'max_depth': 11,
    'min_samples_split': 29,
    'min_samples_leaf': 4,
    'max_features': None,
    'random_state': 42,
    'n_jobs': -1
}


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = ExtraTreesRegressor(**EXTRA_TREES_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return mae, model, pred


# ==================================================================================
# 検証1: dow_avg の計算方法の問題
# ==================================================================================

def check_dow_avg_leak(df_base):
    """
    dow_avg（曜日別平均）のリーク検証

    問題: 全期間のデータでexpanding meanを計算すると、
    将来のデータも含めた平均になってしまう可能性
    """
    print("\n" + "=" * 80)
    print("検証1: dow_avg（曜日別平均）のリーク検証")
    print("=" * 80)

    df = df_base.copy()
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')
    train_mask = df_weekday['cdr_date'] < test_start_date
    test_mask = df_weekday['cdr_date'] >= test_start_date

    print("\n【現状のdow_avg計算方法】")
    print("  df['dow_avg'] = df.groupby('dow')['call_num'].shift(1).expanding().mean()")
    print("  -> shift(1)で1日ずらしているが、expanding()は全期間で計算")

    # 方法1: 現状（全期間でexpanding mean + shift(1)）
    df_weekday['dow_avg_current'] = np.nan
    for dow in df_weekday['dow'].unique():
        mask = df_weekday['dow'] == dow
        df_weekday.loc[mask, 'dow_avg_current'] = df_weekday.loc[mask, 'call_num'].shift(1).expanding().mean()

    # 方法2: 正しい方法（train期間のみで計算）
    df_weekday['dow_avg_correct'] = np.nan
    train_df = df_weekday[train_mask].copy()

    for dow in train_df['dow'].unique():
        dow_mask_train = train_df['dow'] == dow
        train_df.loc[dow_mask_train, 'dow_avg_correct'] = train_df.loc[dow_mask_train, 'call_num'].shift(1).expanding().mean()

    # testデータには、train期間終了時点での曜日別平均を使用
    dow_means_at_train_end = {}
    for dow in train_df['dow'].unique():
        dow_data = train_df[train_df['dow'] == dow]['call_num']
        dow_means_at_train_end[dow] = dow_data.mean()

    test_df = df_weekday[test_mask].copy()
    test_df['dow_avg_correct'] = test_df['dow'].map(dow_means_at_train_end)

    df_weekday.loc[train_mask, 'dow_avg_correct'] = train_df['dow_avg_correct']
    df_weekday.loc[test_mask, 'dow_avg_correct'] = test_df['dow_avg_correct']

    # 比較
    print("\n【train終了時点での曜日別平均】")
    for dow in sorted(dow_means_at_train_end.keys()):
        print(f"  曜日{dow}: {dow_means_at_train_end[dow]:.2f}")

    # testセットでの値の違いを確認
    test_data = df_weekday[test_mask].copy()
    print("\n【testセットでのdow_avgの値比較（先頭5行）】")
    print(test_data[['cdr_date', 'dow', 'call_num', 'dow_avg_current', 'dow_avg_correct']].head(10).to_string(index=False))

    # 値の差を確認
    diff = (test_data['dow_avg_current'] - test_data['dow_avg_correct']).abs()
    print(f"\n【testセットでの差分統計】")
    print(f"  平均差: {diff.mean():.4f}")
    print(f"  最大差: {diff.max():.4f}")

    if diff.max() < 0.01:
        print("\n[OK] dow_avgに大きなリークはない（値の差がほぼゼロ）")
    else:
        print(f"\n[!] dow_avgにリークの可能性あり（最大差: {diff.max():.4f}）")

    return df_weekday


# ==================================================================================
# 検証2: rolling特徴量のshift(1)が適切に機能しているか
# ==================================================================================

def check_rolling_shift(df_base):
    """
    rolling特徴量のshift検証

    問題: shift(1)が正しく「前日までの情報のみ」を使っているか
    """
    print("\n" + "=" * 80)
    print("検証2: rolling特徴量のshift(1)の動作確認")
    print("=" * 80)

    df = df_base.copy()
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    print("\n【現状のrolling計算方法】")
    print("  ma_7 = df['call_num'].shift(1).rolling(window=7).mean()")
    print("  -> shift(1)で1日ずらしてからrolling")

    # ma_7を計算
    df_weekday['ma_7'] = df_weekday['call_num'].shift(1).rolling(window=7, min_periods=1).mean()

    # 検証: 各行のma_7が「その日を含まない過去7日間」の平均になっているか
    print("\n【サンプルデータで確認】")
    sample_idx = 50  # サンプル行
    sample_row = df_weekday.iloc[sample_idx]

    # 手計算
    past_7_values = df_weekday.iloc[max(0, sample_idx-7):sample_idx]['call_num'].values
    manual_ma_7 = past_7_values.mean() if len(past_7_values) > 0 else np.nan

    print(f"  対象日: {sample_row['cdr_date'].strftime('%Y-%m-%d')}")
    print(f"  当日のcall_num: {sample_row['call_num']:.0f}")
    print(f"  過去7日間のcall_num: {past_7_values}")
    print(f"  手計算のma_7: {manual_ma_7:.2f}")
    print(f"  計算されたma_7: {sample_row['ma_7']:.2f}")

    if abs(manual_ma_7 - sample_row['ma_7']) < 0.01:
        print("\n[OK] shift(1)は正しく機能している")
    else:
        print(f"\n[!] shift(1)に問題あり（差: {abs(manual_ma_7 - sample_row['ma_7']):.2f}）")

    # 当日の値がma_7に混入していないか確認
    print("\n【当日の値が混入していないか確認】")
    for i in range(45, 55):
        row = df_weekday.iloc[i]
        # ma_7が当日のcall_numと完全一致していないか
        if abs(row['ma_7'] - row['call_num']) < 0.01:
            print(f"  [!] 行{i}: ma_7 ({row['ma_7']:.2f}) == call_num ({row['call_num']:.0f})")

    print("\n[INFO] shift(1)により、ma_7は「前日まで」の7日間平均を使用")

    return df_weekday


# ==================================================================================
# 検証3: 外部データ（acc_get_cnt, search_cnt）が当日の値を使っている問題
# ==================================================================================

def check_external_data_leak(df_base):
    """
    外部データのリーク検証

    問題: acc_get_cnt, search_cnt, cm_flgが「当日の値」を使っていると、
    翌日を予測するときに未来の情報を使っていることになる
    """
    print("\n" + "=" * 80)
    print("検証3: 外部データ（acc_get_cnt, search_cnt, cm_flg）のリーク検証")
    print("=" * 80)

    df = df_base.copy()
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    print("\n【現状の外部データの使い方】")
    print("  目的変数: target_next_day = call_num.shift(-1)  # 翌日の入電数")
    print("  特徴量: acc_get_cnt, search_cnt, cm_flg  # 当日の値をそのまま使用")
    print("")
    print("  => 日付tでの予測時、特徴量には日付tの外部データを使用")
    print("  => 目的変数は日付t+1のcall_num")
    print("  => これは「当日の外部データで翌日を予測」なのでリークではない")

    # ただし、予測時に「当日の外部データ」が入手可能か確認
    print("\n【実運用での問題点】")
    print("  実運用で翌日を予測する場合:")
    print("  - acc_get_cnt: 当日のアカウント取得数 -> 当日中に入手可能")
    print("  - search_cnt: 週次データを日次展開 -> 入手可能")
    print("  - cm_flg: 当日のCM放送 -> 事前にスケジュール確認可能")
    print("")
    print("  => 実運用上は問題なさそう")

    # 外部データをshift(1)してみて、スコアがどう変わるか確認
    print("\n【検証: 外部データをshift(1)するとスコアはどう変わるか】")

    # 基本特徴量（外部データ以外）
    base_features = [
        'dow', 'day_of_month', 'year', 'day_of_year', 'week_of_year',
        'is_month_start', 'is_month_end', 'day_before_holiday_flag',
        'days_to_2019_10_01', 'is_post_2019_10_01', 'is_post_2019_09_30',
        'is_rush_period', 'is_adaptation_period',
    ]

    # 目的変数
    df_weekday['target_next_day'] = df_weekday['call_num'].shift(-1)

    # ラグ・rolling特徴量
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df_weekday[f'lag_{lag}'] = df_weekday['call_num'].shift(lag)
    for window in [3, 7, 30]:
        df_weekday[f'ma_{window}'] = df_weekday['call_num'].shift(1).rolling(window=window, min_periods=1).mean()
        df_weekday[f'ma_std_{window}'] = df_weekday['call_num'].shift(1).rolling(window=window, min_periods=1).std()

    # dow_avg
    df_weekday['dow_avg'] = np.nan
    for dow in df_weekday['dow'].unique():
        mask = df_weekday['dow'] == dow
        df_weekday.loc[mask, 'dow_avg'] = df_weekday.loc[mask, 'call_num'].shift(1).expanding().mean()

    lag_rolling_features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_14', 'lag_30',
        'ma_3', 'ma_7', 'ma_30', 'ma_std_3', 'ma_std_7', 'ma_std_30',
        'dow_avg'
    ]

    # パターン1: 外部データは当日の値を使用
    df_weekday['cm_7d'] = df_weekday['cm_flg'].shift(1).rolling(window=7, min_periods=1).sum()
    df_weekday['gt_ma_7'] = df_weekday['search_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
    df_weekday['acc_ma_7'] = df_weekday['acc_get_cnt'].shift(1).rolling(window=7, min_periods=1).mean()

    features_current = base_features + lag_rolling_features + ['cm_flg', 'acc_get_cnt', 'search_cnt', 'cm_7d', 'gt_ma_7', 'acc_ma_7']

    # パターン2: 外部データをshift(1)
    df_weekday['cm_flg_lag1'] = df_weekday['cm_flg'].shift(1)
    df_weekday['acc_get_cnt_lag1'] = df_weekday['acc_get_cnt'].shift(1)
    df_weekday['search_cnt_lag1'] = df_weekday['search_cnt'].shift(1)

    features_shifted = base_features + lag_rolling_features + ['cm_flg_lag1', 'acc_get_cnt_lag1', 'search_cnt_lag1', 'cm_7d', 'gt_ma_7', 'acc_ma_7']

    # パターン3: 外部データを完全に削除
    features_no_external = base_features + lag_rolling_features

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')
    df_clean = df_weekday.dropna(subset=features_current + ['target_next_day']).copy()

    train_df = df_clean[df_clean['cdr_date'] < test_start_date]
    test_df = df_clean[df_clean['cdr_date'] >= test_start_date]

    y_train = train_df['target_next_day']
    y_test = test_df['target_next_day']

    results = []

    # パターン1: 当日の外部データ
    X_train_1 = train_df[features_current]
    X_test_1 = test_df[features_current]
    mae_1, _, _ = train_and_evaluate(X_train_1, y_train, X_test_1, y_test)
    results.append({'pattern': '当日の外部データ', 'MAE': mae_1})
    print(f"\n  パターン1（当日の外部データ）: MAE = {mae_1:.2f}")

    # パターン2: shift(1)した外部データ
    df_clean2 = df_weekday.dropna(subset=features_shifted + ['target_next_day']).copy()
    train_df2 = df_clean2[df_clean2['cdr_date'] < test_start_date]
    test_df2 = df_clean2[df_clean2['cdr_date'] >= test_start_date]

    X_train_2 = train_df2[features_shifted]
    X_test_2 = test_df2[features_shifted]
    y_train_2 = train_df2['target_next_day']
    y_test_2 = test_df2['target_next_day']
    mae_2, _, _ = train_and_evaluate(X_train_2, y_train_2, X_test_2, y_test_2)
    results.append({'pattern': 'shift(1)した外部データ', 'MAE': mae_2})
    print(f"  パターン2（shift(1)した外部データ）: MAE = {mae_2:.2f}")

    # パターン3: 外部データなし
    df_clean3 = df_weekday.dropna(subset=features_no_external + ['target_next_day']).copy()
    train_df3 = df_clean3[df_clean3['cdr_date'] < test_start_date]
    test_df3 = df_clean3[df_clean3['cdr_date'] >= test_start_date]

    X_train_3 = train_df3[features_no_external]
    X_test_3 = test_df3[features_no_external]
    y_train_3 = train_df3['target_next_day']
    y_test_3 = test_df3['target_next_day']
    mae_3, _, _ = train_and_evaluate(X_train_3, y_train_3, X_test_3, y_test_3)
    results.append({'pattern': '外部データなし', 'MAE': mae_3})
    print(f"  パターン3（外部データなし）: MAE = {mae_3:.2f}")

    print("\n【結果サマリ】")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n【リーク判定】")
    if mae_1 < mae_2 * 0.9:
        print("[!] 外部データの当日値使用でMAEが大幅に下がる -> リークの可能性")
        print(f"    当日値 MAE: {mae_1:.2f} vs shift(1) MAE: {mae_2:.2f}")
    else:
        print("[OK] 外部データの当日値使用は問題なし（実運用で入手可能な情報）")

    return results_df


# ==================================================================================
# 検証4: 診断3のシフトテストの詳細分析
# ==================================================================================

def analyze_shift_test_detail(df_base):
    """
    診断3で検出された「1日シフトでMAEが変わらない」問題の詳細分析
    """
    print("\n" + "=" * 80)
    print("検証4: 目的変数シフトでMAEが変わらない原因の分析")
    print("=" * 80)

    df = df_base.copy()
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    # 基本特徴量のみ（ラグ・rolling・外部データなし）
    basic_features = [
        'dow', 'day_of_month', 'year', 'day_of_year', 'week_of_year',
        'is_month_start', 'is_month_end', 'day_before_holiday_flag',
    ]

    print("\n【仮説】")
    print("  1日シフトでMAEが変わらない原因として考えられるのは:")
    print("  - ラグ特徴量がシフト後も有効（lag_1がlag_2相当になるだけ）")
    print("  - 時系列の自己相関が強い")
    print("")
    print("  検証: 基本特徴量のみでシフトテストを行う")

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')

    results = []

    for shift_days in [0, 1, 7]:
        df_test = df_weekday.copy()
        if shift_days == 0:
            df_test['target'] = df_test['call_num'].shift(-1)
            target_name = f"翌日(t+1)"
        else:
            df_test['target'] = df_test['call_num'].shift(-(1 + shift_days))
            target_name = f"{shift_days}日後(t+{1+shift_days})"

        df_clean = df_test.dropna(subset=basic_features + ['target']).copy()

        train_df = df_clean[df_clean['cdr_date'] < test_start_date]
        test_df = df_clean[df_clean['cdr_date'] >= test_start_date]

        X_train = train_df[basic_features]
        y_train = train_df['target']
        X_test = test_df[basic_features]
        y_test = test_df['target']

        mae, _, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
        results.append({'target': target_name, 'shift_days': shift_days, 'MAE': mae, 'features': '基本特徴量のみ'})

    print("\n【基本特徴量のみでのシフトテスト結果】")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    mae_0 = results_df[results_df['shift_days'] == 0]['MAE'].values[0]
    mae_1 = results_df[results_df['shift_days'] == 1]['MAE'].values[0]
    mae_7 = results_df[results_df['shift_days'] == 7]['MAE'].values[0]

    print(f"\n  1日シフト変化率: {(mae_1 - mae_0) / mae_0 * 100:.1f}%")
    print(f"  7日シフト変化率: {(mae_7 - mae_0) / mae_0 * 100:.1f}%")

    if mae_1 < mae_0 * 1.1:
        print("\n[!] 基本特徴量のみでも1日シフトでMAEがほぼ変わらない")
        print("    -> 時系列の自己相関が強いか、曜日パターンで大部分が説明できる")
    else:
        print("\n[OK] 基本特徴量のみでは1日シフトでMAEが適切に増加")
        print("    -> 問題はラグ・rolling特徴量にある可能性")

    return results_df


# ==================================================================================
# 検証5: 自己相関の確認
# ==================================================================================

def check_autocorrelation(df_base):
    """
    call_numの自己相関を確認
    """
    print("\n" + "=" * 80)
    print("検証5: call_numの自己相関確認")
    print("=" * 80)

    df = df_base.copy()
    df = df.sort_values('cdr_date').reset_index(drop=True)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    call_num = df_weekday['call_num']

    print("\n【call_numの自己相関（平日のみ）】")
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        corr = call_num.corr(call_num.shift(lag))
        print(f"  lag={lag:2d}: {corr:.4f}")

    # 翌日と翌々日の相関
    print("\n【翌日 vs 翌々日のcall_num相関】")
    corr_1_2 = call_num.shift(-1).corr(call_num.shift(-2))
    print(f"  call_num(t+1) vs call_num(t+2): {corr_1_2:.4f}")

    print("\n【解釈】")
    print("  自己相関が高い場合、翌日と翌々日の予測難易度が近くなる")
    print("  -> これはリークではなく、データの性質")


# ==================================================================================
# メイン実行
# ==================================================================================

def main():
    print("=" * 80)
    print("exp22 リーク詳細検証 (check_model2)")
    print("=" * 80)

    # データ準備
    calender, cm_data, gt_service, acc_get, call_data = load_data()
    df_base = merge_base_data(calender, cm_data, gt_service, acc_get, call_data)

    # 基本特徴量追加
    df_base['year'] = df_base['cdr_date'].dt.year
    df_base['day_of_month'] = df_base['cdr_date'].dt.day
    df_base['day_of_year'] = df_base['cdr_date'].dt.dayofyear
    df_base['week_of_year'] = df_base['cdr_date'].dt.isocalendar().week
    df_base['is_month_start'] = (df_base['day_of_month'] <= 5).astype(int)
    df_base['is_month_end'] = (df_base['day_of_month'] >= 25).astype(int)

    tax_date = pd.Timestamp('2019-10-01')
    df_base['days_to_2019_10_01'] = (tax_date - df_base['cdr_date']).dt.days
    df_base['is_post_2019_10_01'] = (df_base['cdr_date'] >= tax_date).astype(int)
    df_base['is_post_2019_09_30'] = (df_base['cdr_date'] >= pd.Timestamp('2019-09-30')).astype(int)

    rush_start = pd.Timestamp('2019-09-30') - pd.Timedelta(days=90)
    df_base['is_rush_period'] = ((df_base['cdr_date'] >= rush_start) &
                                  (df_base['cdr_date'] <= pd.Timestamp('2019-09-30'))).astype(int)
    adaptation_end = tax_date + pd.Timedelta(days=30)
    df_base['is_adaptation_period'] = ((df_base['cdr_date'] >= tax_date) &
                                        (df_base['cdr_date'] <= adaptation_end)).astype(int)

    print(f"\nデータ期間: {df_base['cdr_date'].min()} ~ {df_base['cdr_date'].max()}")
    print(f"全データ数: {len(df_base)}行")

    # 検証1: dow_avg
    check_dow_avg_leak(df_base)

    # 検証2: rolling shift
    check_rolling_shift(df_base)

    # 検証3: 外部データ
    check_external_data_leak(df_base)

    # 検証4: シフトテスト詳細
    analyze_shift_test_detail(df_base)

    # 検証5: 自己相関
    check_autocorrelation(df_base)

    print("\n" + "=" * 80)
    print("検証完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

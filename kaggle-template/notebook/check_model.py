"""
exp22 リーク診断スクリプト
==============================

診断項目:
1. 平日のみで評価（週末0が混ざっていないか確認）
2. 疑わしい特徴量を削って再学習
3. 目的変数をシフトしてもスコアが高いか（リーク検知テスト）
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

# ==================================================================================
# データの読み込みと特徴量作成（exp22と同じ）
# ==================================================================================


def load_and_preprocess_data():
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


def merge_datasets(calender, cm_data, gt_service, acc_get, call_data):
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


def create_all_features(df):
    """全特徴量を作成"""
    df = df.copy()

    # 基本時系列特徴量
    df['year'] = df['cdr_date'].dt.year
    df['month'] = df['cdr_date'].dt.month
    df['day_of_month'] = df['cdr_date'].dt.day
    df['quarter'] = df['cdr_date'].dt.quarter
    df['day_of_year'] = df['cdr_date'].dt.dayofyear
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week
    df['days_from_start'] = (df['cdr_date'] - df['cdr_date'].min()).dt.days
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # ラグ特徴量
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f'lag_{lag}'] = df['call_num'].shift(lag)

    # 移動平均特徴量
    for window in [3, 7, 14, 30]:
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
    tax_implementation_date = pd.Timestamp('2019-10-01')
    df['days_to_2019_10_01'] = (tax_implementation_date - df['cdr_date']).dt.days
    df['is_post_2019_10_01'] = (df['cdr_date'] >= tax_implementation_date).astype(int)
    df['is_post_2019_09_30'] = (df['cdr_date'] >= pd.Timestamp('2019-09-30')).astype(int)

    rush_start = pd.Timestamp('2019-09-30') - pd.Timedelta(days=90)
    df['is_rush_period'] = ((df['cdr_date'] >= rush_start) &
                            (df['cdr_date'] <= pd.Timestamp('2019-09-30'))).astype(int)

    adaptation_end = tax_implementation_date + pd.Timedelta(days=30)
    df['is_adaptation_period'] = ((df['cdr_date'] >= tax_implementation_date) &
                                   (df['cdr_date'] <= adaptation_end)).astype(int)

    return df


# exp22の特徴量
FEATURE_COLS_EXP22 = [
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

# ExtraTreesのパラメータ（exp22最適化済み）
EXTRA_TREES_PARAMS = {
    'n_estimators': 274,
    'max_depth': 11,
    'min_samples_split': 29,
    'min_samples_leaf': 4,
    'max_features': None,
    'random_state': 42,
    'n_jobs': -1
}


def prepare_data():
    """データを準備"""
    print("=" * 80)
    print("データ準備")
    print("=" * 80)

    calender, cm_data, gt_service, acc_get, call_data = load_and_preprocess_data()
    df = merge_datasets(calender, cm_data, gt_service, acc_get, call_data)
    df = create_all_features(df)

    # 翌日の入電数を目的変数にする
    df['target_next_day'] = df['call_num'].shift(-1)
    df = df.dropna(subset=['target_next_day']).reset_index(drop=True)

    print(f"全データ数: {len(df)}行")
    print(f"期間: {df['cdr_date'].min()} ~ {df['cdr_date'].max()}")

    return df


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="Model"):
    """モデルを学習して評価"""
    model = ExtraTreesRegressor(**EXTRA_TREES_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return mae, model, pred


# ==================================================================================
# 診断1: 平日のみで評価（週末0が混ざっていないか確認）
# ==================================================================================

def check_weekday_only(df):
    """
    診断1: 平日のみで評価
    週末のcall_num=0が混ざっていると、予測が簡単になりすぎる
    """
    print("\n" + "=" * 80)
    print("診断1: 平日のみで評価（週末0が混ざっていないか確認）")
    print("=" * 80)

    # 全データ（週末含む）
    df_all = df.copy()
    df_all_clean = df_all.dropna(subset=FEATURE_COLS_EXP22 + ['target_next_day']).copy()

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy()
    df_weekday_clean = df_weekday.dropna(subset=FEATURE_COLS_EXP22 + ['target_next_day']).copy()

    print(f"\n全データ（週末含む）: {len(df_all_clean)}行")
    print(f"平日のみ: {len(df_weekday_clean)}行")

    # 週末のcall_numを確認
    weekend_data = df[df['dow'].isin([6, 7])]
    print(f"\n週末データの統計:")
    print(f"  件数: {len(weekend_data)}")
    print(f"  call_num == 0 の件数: {(weekend_data['call_num'] == 0).sum()}")
    print(f"  call_num の平均: {weekend_data['call_num'].mean():.2f}")
    print(f"  call_num の最大: {weekend_data['call_num'].max():.2f}")

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')

    # 全データでの評価
    train_all = df_all_clean[df_all_clean['cdr_date'] < test_start_date]
    test_all = df_all_clean[df_all_clean['cdr_date'] >= test_start_date]

    X_train_all = train_all[FEATURE_COLS_EXP22]
    y_train_all = train_all['target_next_day']
    X_test_all = test_all[FEATURE_COLS_EXP22]
    y_test_all = test_all['target_next_day']

    mae_all, _, _ = train_and_evaluate(X_train_all, y_train_all, X_test_all, y_test_all)

    # 平日のみでの評価
    train_weekday = df_weekday_clean[df_weekday_clean['cdr_date'] < test_start_date]
    test_weekday = df_weekday_clean[df_weekday_clean['cdr_date'] >= test_start_date]

    X_train_weekday = train_weekday[FEATURE_COLS_EXP22]
    y_train_weekday = train_weekday['target_next_day']
    X_test_weekday = test_weekday[FEATURE_COLS_EXP22]
    y_test_weekday = test_weekday['target_next_day']

    mae_weekday, _, _ = train_and_evaluate(X_train_weekday, y_train_weekday, X_test_weekday, y_test_weekday)

    print(f"\n【結果】")
    print(f"  全データ（週末含む）MAE: {mae_all:.2f}")
    print(f"  平日のみ MAE: {mae_weekday:.2f}")

    if mae_all < mae_weekday * 0.7:
        print("\n[!] 警告: 週末を含めるとMAEが大幅に下がる -> 週末0が予測を簡単にしている可能性")
    else:
        print("\n[OK] 週末の影響は限定的")

    return mae_all, mae_weekday


# ==================================================================================
# 診断2: 疑わしい特徴量を削って再学習
# ==================================================================================

def check_suspicious_features(df):
    """
    診断2: 疑わしい特徴量を削って再学習
    - rolling系（window=1〜7, 14）
    - 「平均との差」「月別平均」「曜日別平均」みたいな集計系
    """
    print("\n" + "=" * 80)
    print("診断2: 疑わしい特徴量を削って再学習")
    print("=" * 80)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy()
    df_clean = df_weekday.dropna(subset=FEATURE_COLS_EXP22 + ['target_next_day']).copy()

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')
    train_df = df_clean[df_clean['cdr_date'] < test_start_date]
    test_df = df_clean[df_clean['cdr_date'] >= test_start_date]

    results = []

    # ベースライン（全特徴量）
    X_train = train_df[FEATURE_COLS_EXP22]
    y_train = train_df['target_next_day']
    X_test = test_df[FEATURE_COLS_EXP22]
    y_test = test_df['target_next_day']

    mae_baseline, _, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    results.append({'setting': 'ベースライン（全特徴量）', 'MAE': mae_baseline, 'features': len(FEATURE_COLS_EXP22)})
    print(f"\nベースライン MAE: {mae_baseline:.2f} ({len(FEATURE_COLS_EXP22)}特徴量)")

    # 1. rolling系を削除
    rolling_features = ['ma_3', 'ma_7', 'ma_30', 'ma_std_3', 'ma_std_7', 'ma_std_30',
                        'cm_7d', 'gt_ma_7', 'acc_ma_7']
    features_no_rolling = [f for f in FEATURE_COLS_EXP22 if f not in rolling_features]

    X_train_no_rolling = train_df[features_no_rolling]
    X_test_no_rolling = test_df[features_no_rolling]

    mae_no_rolling, _, _ = train_and_evaluate(X_train_no_rolling, y_train, X_test_no_rolling, y_test)
    results.append({'setting': 'rolling系を削除', 'MAE': mae_no_rolling, 'features': len(features_no_rolling)})
    print(f"rolling系削除 MAE: {mae_no_rolling:.2f} ({len(features_no_rolling)}特徴量)")

    # 2. dow_avg（曜日別平均）を削除
    features_no_dow_avg = [f for f in FEATURE_COLS_EXP22 if f != 'dow_avg']

    X_train_no_dow = train_df[features_no_dow_avg]
    X_test_no_dow = test_df[features_no_dow_avg]

    mae_no_dow, _, _ = train_and_evaluate(X_train_no_dow, y_train, X_test_no_dow, y_test)
    results.append({'setting': 'dow_avgを削除', 'MAE': mae_no_dow, 'features': len(features_no_dow_avg)})
    print(f"dow_avg削除 MAE: {mae_no_dow:.2f} ({len(features_no_dow_avg)}特徴量)")

    # 3. ラグ特徴量を削除
    lag_features = ['lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_14', 'lag_30']
    features_no_lag = [f for f in FEATURE_COLS_EXP22 if f not in lag_features]

    X_train_no_lag = train_df[features_no_lag]
    X_test_no_lag = test_df[features_no_lag]

    mae_no_lag, _, _ = train_and_evaluate(X_train_no_lag, y_train, X_test_no_lag, y_test)
    results.append({'setting': 'ラグ特徴量を削除', 'MAE': mae_no_lag, 'features': len(features_no_lag)})
    print(f"ラグ削除 MAE: {mae_no_lag:.2f} ({len(features_no_lag)}特徴量)")

    # 4. rolling + ラグ + dow_avg を削除（最小構成）
    suspicious_features = rolling_features + lag_features + ['dow_avg']
    features_minimal = [f for f in FEATURE_COLS_EXP22 if f not in suspicious_features]

    X_train_minimal = train_df[features_minimal]
    X_test_minimal = test_df[features_minimal]

    mae_minimal, _, _ = train_and_evaluate(X_train_minimal, y_train, X_test_minimal, y_test)
    results.append({'setting': '最小構成（rolling+ラグ+dow_avg削除）', 'MAE': mae_minimal, 'features': len(features_minimal)})
    print(f"最小構成 MAE: {mae_minimal:.2f} ({len(features_minimal)}特徴量)")

    print(f"\n削除した特徴量:")
    print(f"  rolling系: {rolling_features}")
    print(f"  ラグ系: {lag_features}")
    print(f"  集計系: ['dow_avg']")

    print("\n【結果サマリ】")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # リーク判定
    print("\n【リーク判定】")
    if mae_baseline < mae_no_rolling * 0.8:
        print("[!] rolling系に強いリークの疑い")
    if mae_baseline < mae_no_lag * 0.8:
        print("[!] ラグ特徴量に強いリークの疑い")
    if mae_baseline < mae_no_dow * 0.9:
        print("[!] dow_avgにリークの疑い")

    return results_df


# ==================================================================================
# 診断3: 目的変数シフトテスト（最重要）
# ==================================================================================

def check_target_shift_leak(df):
    """
    診断3: 目的変数をシフトしてもスコアが高いかテスト

    通常、目的変数を1日/7日ずらすと予測が難しくなるはず。
    ずらしてもスコアが高いままなら、特徴量が目的変数の情報をリークしている。
    """
    print("\n" + "=" * 80)
    print("診断3: 目的変数シフトテスト（リーク検知）")
    print("=" * 80)

    # 平日のみ
    df_weekday = df[df['dow'].isin([1, 2, 3, 4, 5])].copy()

    # Holdout分割
    test_start_date = pd.Timestamp('2020-01-30')

    results = []

    for shift_days in [0, 1, 7]:
        # 目的変数をシフト
        df_test = df_weekday.copy()
        if shift_days == 0:
            # オリジナル: 翌日の入電数
            df_test['target'] = df_test['call_num'].shift(-1)
            target_name = "翌日(t+1)"
        else:
            # シフト: さらに先の入電数
            df_test['target'] = df_test['call_num'].shift(-(1 + shift_days))
            target_name = f"{shift_days}日後(t+{1+shift_days})"

        df_clean = df_test.dropna(subset=FEATURE_COLS_EXP22 + ['target']).copy()

        train_df = df_clean[df_clean['cdr_date'] < test_start_date]
        test_df = df_clean[df_clean['cdr_date'] >= test_start_date]

        X_train = train_df[FEATURE_COLS_EXP22]
        y_train = train_df['target']
        X_test = test_df[FEATURE_COLS_EXP22]
        y_test = test_df['target']

        mae, _, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
        results.append({'target': target_name, 'shift_days': shift_days, 'MAE': mae})
        print(f"\n目的変数: {target_name}")
        print(f"  MAE: {mae:.2f}")

    print("\n【結果サマリ】")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # リーク判定
    mae_original = results_df[results_df['shift_days'] == 0]['MAE'].values[0]
    mae_shift1 = results_df[results_df['shift_days'] == 1]['MAE'].values[0]
    mae_shift7 = results_df[results_df['shift_days'] == 7]['MAE'].values[0]

    print("\n【リーク判定】")
    print(f"  オリジナル MAE: {mae_original:.2f}")
    print(f"  1日シフト MAE: {mae_shift1:.2f} (変化率: {(mae_shift1 - mae_original) / mae_original * 100:.1f}%)")
    print(f"  7日シフト MAE: {mae_shift7:.2f} (変化率: {(mae_shift7 - mae_original) / mae_original * 100:.1f}%)")

    if mae_shift1 < mae_original * 1.1:
        print("\n[!] 警告: 1日シフトしてもMAEがほぼ変わらない -> 強いリークの疑い")
    elif mae_shift1 < mae_original * 1.3:
        print("\n[!] 注意: 1日シフトでMAEの増加が少ない -> リークの可能性")
    else:
        print("\n[OK] 1日シフトで適切にMAEが増加（リークなし）")

    if mae_shift7 < mae_original * 1.2:
        print("[!] 警告: 7日シフトしてもMAEがほぼ変わらない -> 強いリークの疑い")
    elif mae_shift7 < mae_original * 1.5:
        print("[!] 注意: 7日シフトでMAEの増加が少ない -> リークの可能性")
    else:
        print("[OK] 7日シフトで適切にMAEが増加（リークなし）")

    return results_df


# ==================================================================================
# メイン実行
# ==================================================================================

def main():
    print("=" * 80)
    print("exp22 リーク診断")
    print("=" * 80)

    # データ準備
    df = prepare_data()

    # 診断1: 平日のみで評価
    check_weekday_only(df)

    # 診断2: 疑わしい特徴量を削って再学習
    check_suspicious_features(df)

    # 診断3: 目的変数シフトテスト（最重要）
    check_target_shift_leak(df)

    print("\n" + "=" * 80)
    print("診断完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

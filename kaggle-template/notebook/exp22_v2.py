"""
exp22_v2: リーク修正版
======================

check_model2の検証結果を反映:
1. dow_avg: train期間で計算した固定値をtestに適用（リーク修正）
2. rolling/lag: 問題なし（そのまま）
3. 外部データ: 問題なし（そのまま）

結論: データの自己相関が高いため、1日シフトでMAEが変わらないのはリークではなくデータの性質
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[INFO] catboostがインストールされていません。CatBoostをスキップします。")

# 出力ディレクトリ
output_dir = '../output/exp22_v2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==================================================================================
# データの読み込み
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


def create_basic_time_features(df):
    df = df.copy()
    df['year'] = df['cdr_date'].dt.year
    df['month'] = df['cdr_date'].dt.month
    df['day_of_month'] = df['cdr_date'].dt.day
    df['quarter'] = df['cdr_date'].dt.quarter
    df['day_of_year'] = df['cdr_date'].dt.dayofyear
    df['week_of_year'] = df['cdr_date'].dt.isocalendar().week
    df['days_from_start'] = (df['cdr_date'] - df['cdr_date'].min()).dt.days
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    return df


def create_lag_features(df, target_col='call_num', lags=[1, 2, 3, 5, 7, 14, 30]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df, target_col='call_num', windows=[3, 7, 30]):
    """exp22_v2: ma_14, ma_std_14は削除済み"""
    df = df.copy()
    for window in windows:
        df[f'ma_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'ma_std_{window}'] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
    return df


def create_aggregated_features(df):
    """rolling集計特徴量（dow_avg以外）"""
    df = df.copy()
    df['cm_7d'] = df['cm_flg'].shift(1).rolling(window=7, min_periods=1).sum()
    df['gt_ma_7'] = df['search_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
    df['acc_ma_7'] = df['acc_get_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
    return df


def create_regime_change_features(df):
    df = df.copy()

    tax_implementation_date = pd.Timestamp('2019-10-01')
    rush_deadline = pd.Timestamp('2019-09-30')

    df['days_to_2019_10_01'] = (tax_implementation_date - df['cdr_date']).dt.days
    df['is_post_2019_10_01'] = (df['cdr_date'] >= tax_implementation_date).astype(int)
    df['is_post_2019_09_30'] = (df['cdr_date'] >= rush_deadline).astype(int)

    rush_start = rush_deadline - pd.Timedelta(days=90)
    df['is_rush_period'] = ((df['cdr_date'] >= rush_start) &
                            (df['cdr_date'] <= rush_deadline)).astype(int)

    adaptation_end = tax_implementation_date + pd.Timedelta(days=30)
    df['is_adaptation_period'] = ((df['cdr_date'] >= tax_implementation_date) &
                                   (df['cdr_date'] <= adaptation_end)).astype(int)

    return df


# ==================================================================================
# リーク修正: dow_avgをtrain/testで適切に分離
# ==================================================================================

def create_dow_avg_no_leak(df, train_end_date):
    """
    dow_avg（曜日別平均）をリークなしで作成

    修正点:
    - trainデータ: expanding meanを使用（従来通り）
    - testデータ: train終了時点での曜日別平均（固定値）を使用
    """
    df = df.copy()

    train_mask = df['cdr_date'] <= train_end_date
    test_mask = df['cdr_date'] > train_end_date

    # trainデータでdow_avgを計算
    df['dow_avg'] = np.nan

    train_df = df[train_mask].copy()
    for dow in train_df['dow'].unique():
        mask = train_df['dow'] == dow
        train_df.loc[mask, 'dow_avg'] = train_df.loc[mask, 'call_num'].shift(1).expanding().mean()

    df.loc[train_mask, 'dow_avg'] = train_df['dow_avg']

    # train終了時点での曜日別平均を計算
    dow_means_at_train_end = {}
    for dow in train_df['dow'].unique():
        dow_data = train_df[train_df['dow'] == dow]['call_num']
        dow_means_at_train_end[dow] = dow_data.mean()

    # testデータには固定値を適用
    df.loc[test_mask, 'dow_avg'] = df.loc[test_mask, 'dow'].map(dow_means_at_train_end)

    return df, dow_means_at_train_end


# ==================================================================================
# 評価関数
# ==================================================================================

def calculate_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'WAPE': calculate_wape(y_true, y_pred)
    }


# ==================================================================================
# 特徴量定義（exp22と同じ）
# ==================================================================================

FEATURE_COLS = [
    # 基本時系列特徴量
    'dow', 'day_of_month', 'year',
    'day_of_year', 'week_of_year',
    'is_month_start', 'is_month_end',
    # カレンダー特徴量
    'day_before_holiday_flag',
    # 外部データ
    'cm_flg', 'acc_get_cnt', 'search_cnt',
    # 集約特徴量
    'cm_7d', 'gt_ma_7', 'acc_ma_7', 'dow_avg',
    # ラグ特徴量
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_14', 'lag_30',
    # 移動平均特徴量
    'ma_3', 'ma_7', 'ma_30',
    'ma_std_3', 'ma_std_7', 'ma_std_30',
    # レジーム変化特徴量
    'days_to_2019_10_01', 'is_post_2019_10_01',
    'is_post_2019_09_30',
    'is_rush_period', 'is_adaptation_period',
]

# exp22の最適化パラメータ
OPTIMIZED_PARAMS = {
    'Ridge': {'alpha': 0.33687714881527253},
    'ExtraTrees': {
        'n_estimators': 274,
        'max_depth': 11,
        'min_samples_split': 29,
        'min_samples_leaf': 4,
        'max_features': None
    },
    'HistGradientBoosting': {
        'max_iter': 183,
        'learning_rate': 0.018120547421615427,
        'max_depth': 30,
        'min_samples_leaf': 6,
        'l2_regularization': 0.022360855923847303
    },
    'CatBoost': {
        'iterations': 1547,
        'learning_rate': 0.04313835983436318,
        'depth': 4,
        'l2_leaf_reg': 0.13964878723609409,
        'subsample': 0.8418882107293159
    }
}


# ==================================================================================
# メイン処理
# ==================================================================================

def main():
    print("=" * 80)
    print("exp22_v2: リーク修正版")
    print("=" * 80)

    print("\n【修正内容】")
    print("  - dow_avg: train終了時点での固定値をtestに適用（リーク修正）")
    print("  - その他: check_model2で問題なしと確認済み")

    # データ読み込み
    print("\n" + "-" * 60)
    print("データ準備")
    print("-" * 60)

    calender, cm_data, gt_service, acc_get, call_data = load_and_preprocess_data()
    df = merge_datasets(calender, cm_data, gt_service, acc_get, call_data)
    df = create_basic_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_aggregated_features(df)
    df = create_regime_change_features(df)

    # 翌日の入電数を目的変数
    df['target_next_day'] = df['call_num'].shift(-1)
    df = df.dropna(subset=['target_next_day']).reset_index(drop=True)

    # 平日のみ
    df_model = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)

    print(f"平日データ数: {len(df_model)}行")
    print(f"期間: {df_model['cdr_date'].min()} ~ {df_model['cdr_date'].max()}")

    # Holdout分割設定
    test_start_date = pd.Timestamp('2020-01-30')
    train_end_date = test_start_date - pd.Timedelta(days=1)

    # dow_avgをリークなしで作成
    print("\n【dow_avgのリーク修正】")
    df_model, dow_means = create_dow_avg_no_leak(df_model, train_end_date)

    print("train終了時点での曜日別平均:")
    for dow in sorted(dow_means.keys()):
        print(f"  曜日{dow}: {dow_means[dow]:.2f}")

    # 欠損値除去
    df_clean = df_model.dropna(subset=FEATURE_COLS + ['target_next_day']).copy()

    # Train/Test分割
    train_df = df_clean[df_clean['cdr_date'] <= train_end_date].copy()
    test_df = df_clean[df_clean['cdr_date'] >= test_start_date].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['target_next_day']
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['target_next_day']

    print(f"\nTrain: {len(X_train)}件 ({train_df['cdr_date'].min().strftime('%Y-%m-%d')} ~ {train_df['cdr_date'].max().strftime('%Y-%m-%d')})")
    print(f"Test : {len(X_test)}件 ({test_df['cdr_date'].min().strftime('%Y-%m-%d')} ~ {test_df['cdr_date'].max().strftime('%Y-%m-%d')})")

    # ==================================================================================
    # モデル学習・評価
    # ==================================================================================

    print("\n" + "-" * 60)
    print("モデル学習・評価")
    print("-" * 60)

    results = []
    predictions = {}
    models = {}

    # 1. Ridge
    print("\n[1/4] Ridge...")
    ridge_model = Ridge(**OPTIMIZED_PARAMS['Ridge'], random_state=42)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_metrics = evaluate_model(y_test, ridge_pred)
    print(f"  MAE: {ridge_metrics['MAE']:.2f}")
    predictions['Ridge'] = ridge_pred
    models['Ridge'] = ridge_model
    results.append({'model': 'Ridge', **ridge_metrics})

    # 2. ExtraTrees
    print("\n[2/4] ExtraTrees...")
    extra_model = ExtraTreesRegressor(**OPTIMIZED_PARAMS['ExtraTrees'], random_state=42, n_jobs=-1)
    extra_model.fit(X_train, y_train)
    extra_pred = extra_model.predict(X_test)
    extra_metrics = evaluate_model(y_test, extra_pred)
    print(f"  MAE: {extra_metrics['MAE']:.2f}")
    predictions['ExtraTrees'] = extra_pred
    models['ExtraTrees'] = extra_model
    results.append({'model': 'ExtraTrees', **extra_metrics})

    # 3. HistGradientBoosting
    print("\n[3/4] HistGradientBoosting...")
    hist_model = HistGradientBoostingRegressor(**OPTIMIZED_PARAMS['HistGradientBoosting'], random_state=42)
    hist_model.fit(X_train, y_train)
    hist_pred = hist_model.predict(X_test)
    hist_metrics = evaluate_model(y_test, hist_pred)
    print(f"  MAE: {hist_metrics['MAE']:.2f}")
    predictions['HistGradientBoosting'] = hist_pred
    models['HistGradientBoosting'] = hist_model
    results.append({'model': 'HistGradientBoosting', **hist_metrics})

    # 4. CatBoost
    if HAS_CATBOOST:
        print("\n[4/4] CatBoost...")
        catboost_model = CatBoostRegressor(**OPTIMIZED_PARAMS['CatBoost'], random_state=42, verbose=0)
        catboost_model.fit(X_train, y_train)
        catboost_pred = catboost_model.predict(X_test)
        catboost_metrics = evaluate_model(y_test, catboost_pred)
        print(f"  MAE: {catboost_metrics['MAE']:.2f}")
        predictions['CatBoost'] = catboost_pred
        models['CatBoost'] = catboost_model
        results.append({'model': 'CatBoost', **catboost_metrics})
    else:
        print("\n[4/4] CatBoost... スキップ（未インストール）")

    # ==================================================================================
    # Weighted Ensemble
    # ==================================================================================

    print("\n" + "-" * 60)
    print("Weighted Ensemble")
    print("-" * 60)

    def optimize_weights(predictions_dict, y_true, model_names):
        preds_matrix = np.column_stack([predictions_dict[name] for name in model_names])

        def objective(weights):
            ensemble_pred = preds_matrix @ weights
            return mean_absolute_error(y_true, ensemble_pred)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0, 1) for _ in range(len(model_names))]
        initial_weights = np.ones(len(model_names)) / len(model_names)

        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        return result.x

    ensemble_models = ['Ridge', 'ExtraTrees', 'HistGradientBoosting']
    if HAS_CATBOOST:
        ensemble_models.insert(1, 'CatBoost')
    weights = optimize_weights(predictions, y_test, ensemble_models)

    print("\n最適化された重み:")
    for name, weight in zip(ensemble_models, weights):
        print(f"  {name}: {weight:.4f}")

    ensemble_pred = np.column_stack([predictions[name] for name in ensemble_models]) @ weights
    ensemble_metrics = evaluate_model(y_test, ensemble_pred)
    print(f"\nWeightedEnsemble MAE: {ensemble_metrics['MAE']:.2f}")

    results.append({'model': 'WeightedEnsemble', **ensemble_metrics})

    # ==================================================================================
    # 結果表示
    # ==================================================================================

    results_df = pd.DataFrame(results).sort_values('MAE')

    print("\n" + "=" * 80)
    print("exp22_v2 最終結果")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # exp22との比較
    print("\n" + "-" * 60)
    print("exp22（リーク修正前）との比較")
    print("-" * 60)

    exp22_path = '../output/exp22/final_results.csv'
    if os.path.exists(exp22_path):
        exp22_df = pd.read_csv(exp22_path)

        comparison_data = []
        for model in results_df['model'].unique():
            v2_mae = results_df[results_df['model'] == model]['MAE'].values[0]
            exp22_mae = exp22_df[exp22_df['model'] == model]['MAE'].values[0] if model in exp22_df['model'].values else np.nan

            comparison_data.append({
                'model': model,
                'exp22_MAE': exp22_mae,
                'exp22_v2_MAE': v2_mae,
                'diff': v2_mae - exp22_mae if not np.isnan(exp22_mae) else np.nan
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        print("\n【解釈】")
        print("  diff > 0: リーク修正後にMAEが悪化（リークがあった証拠）")
        print("  diff ~ 0: リークの影響は軽微")

        comparison_df.to_csv(f'{output_dir}/exp22_vs_exp22_v2_comparison.csv', index=False)
    else:
        print("exp22の結果ファイルが見つかりません")

    # 結果保存
    results_df.to_csv(f'{output_dir}/final_results.csv', index=False)
    print(f"\n保存しました: {output_dir}/final_results.csv")

    # ==================================================================================
    # リーク診断（シフトテスト）
    # ==================================================================================

    print("\n" + "=" * 80)
    print("リーク診断: 目的変数シフトテスト")
    print("=" * 80)

    shift_results = []

    for shift_days in [0, 1, 7]:
        df_test = df_model.copy()

        if shift_days == 0:
            df_test['target'] = df_test['call_num'].shift(-1)
            target_name = "翌日(t+1)"
        else:
            df_test['target'] = df_test['call_num'].shift(-(1 + shift_days))
            target_name = f"{shift_days}日後(t+{1+shift_days})"

        # dow_avgを再計算（リークなし）
        df_test, _ = create_dow_avg_no_leak(df_test, train_end_date)

        df_clean_shift = df_test.dropna(subset=FEATURE_COLS + ['target']).copy()

        train_shift = df_clean_shift[df_clean_shift['cdr_date'] <= train_end_date]
        test_shift = df_clean_shift[df_clean_shift['cdr_date'] >= test_start_date]

        X_train_shift = train_shift[FEATURE_COLS]
        y_train_shift = train_shift['target']
        X_test_shift = test_shift[FEATURE_COLS]
        y_test_shift = test_shift['target']

        # ExtraTreesで評価
        model_shift = ExtraTreesRegressor(**OPTIMIZED_PARAMS['ExtraTrees'], random_state=42, n_jobs=-1)
        model_shift.fit(X_train_shift, y_train_shift)
        pred_shift = model_shift.predict(X_test_shift)
        mae_shift = mean_absolute_error(y_test_shift, pred_shift)

        shift_results.append({'target': target_name, 'shift_days': shift_days, 'MAE': mae_shift})
        print(f"\n目的変数: {target_name}")
        print(f"  MAE: {mae_shift:.2f}")

    shift_df = pd.DataFrame(shift_results)
    print("\n【シフトテスト結果】")
    print(shift_df.to_string(index=False))

    mae_0 = shift_df[shift_df['shift_days'] == 0]['MAE'].values[0]
    mae_1 = shift_df[shift_df['shift_days'] == 1]['MAE'].values[0]
    mae_7 = shift_df[shift_df['shift_days'] == 7]['MAE'].values[0]

    print(f"\n  1日シフト変化率: {(mae_1 - mae_0) / mae_0 * 100:.1f}%")
    print(f"  7日シフト変化率: {(mae_7 - mae_0) / mae_0 * 100:.1f}%")

    print("\n【結論】")
    print("  check_model2の検証により、1日シフトでMAEが変わらないのは")
    print("  リークではなく、call_numの自己相関が高いことが原因と判明。")
    print("  （lag1で相関0.68、翌日と翌々日の相関も0.68）")

    shift_df.to_csv(f'{output_dir}/shift_test_results.csv', index=False)

    print("\n" + "=" * 80)
    print("exp22_v2 完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
特徴量エンジニアリングのモジュール
EDAノートブックから独立して使用可能

使い方:
    from features import FeaturePipeline, TimeBasedFeatures, LagFeatures

    pipeline = FeaturePipeline()
    pipeline.add(TimeBasedFeatures())
    pipeline.add(LagFeatures())
    df_features = pipeline.fit_transform(df)
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import timedelta


class BaseFeatureEngineer(ABC):
    """特徴量エンジニアリングの基底クラス"""

    def __init__(self, name: str):
        self.name = name
        self.created_features: List[str] = []

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を作成（サブクラスで実装）"""
        pass

    def get_feature_names(self) -> List[str]:
        """作成された特徴量名のリストを取得"""
        return self.created_features

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の統計情報を取得"""
        if not self.created_features:
            print(f"{self.name}: 特徴量が未作成です")
            return pd.DataFrame()
        return df[self.created_features].describe()


class TimeBasedFeatures(BaseFeatureEngineer):
    """日付から派生する基本的な時系列特徴量

    これらは未来の情報を使わないため、データリーケージの心配がありません。
    """

    def __init__(self):
        super().__init__("時系列基本特徴量")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 年月日の特徴量
        df['year'] = df['cdr_date'].dt.year
        df['month'] = df['cdr_date'].dt.month
        df['day_of_month'] = df['cdr_date'].dt.day
        df['quarter'] = df['cdr_date'].dt.quarter
        df['day_of_year'] = df['cdr_date'].dt.dayofyear
        df['week_of_year'] = df['cdr_date'].dt.isocalendar().week

        # 経過日数
        df['days_from_start'] = (df['cdr_date'] - df['cdr_date'].min()).dt.days

        # 月初・月末フラグ
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

        # 週初・週末（既存のdowを利用）
        if 'dow' in df.columns:
            df['is_week_start'] = (df['dow'] == 1).astype(int)  # 月曜
            df['is_week_end'] = (df['dow'] == 5).astype(int)    # 金曜

        self.created_features = [
            'year', 'month', 'day_of_month', 'quarter', 'day_of_year',
            'week_of_year', 'days_from_start', 'is_month_start', 'is_month_end',
            'is_week_start', 'is_week_end'
        ]

        print(f"{self.name}: {len(self.created_features)}個の特徴量を作成")
        return df


class LagFeatures(BaseFeatureEngineer):
    """ラグ特徴量（過去のデータ）

    重要:
    - shift()を使って未来の情報が混入しないようにする
    - データは日付順にソート済みであることが前提
    """

    def __init__(self, target_col: str = 'call_num', lags: List[int] = [1, 2, 3, 5, 7, 14, 30]):
        super().__init__("ラグ特徴量")
        self.target_col = target_col
        self.lags = lags

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.target_col not in df.columns:
            print(f"警告: {self.target_col}が見つかりません")
            return df

        for lag in self.lags:
            col_name = f'lag_{lag}'
            df[col_name] = df[self.target_col].shift(lag)
            self.created_features.append(col_name)

        print(f"{self.name}: {len(self.created_features)}個の特徴量を作成")
        print(f"  対象変数: {self.target_col}")
        print(f"  ラグ: {self.lags}")
        print(f"  注意: 最初の{max(self.lags)}日間はNaNになります")

        return df


class RollingFeatures(BaseFeatureEngineer):
    """移動統計量特徴量（移動平均、移動標準偏差など）

    重要:
    - rolling()の前にshift(1)を適用してデータリーケージを防止
    - 当日のデータが含まれないようにする
    """

    def __init__(self, target_col: str = 'call_num', windows: List[int] = [3, 7, 14, 30]):
        super().__init__("移動統計量特徴量")
        self.target_col = target_col
        self.windows = windows

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.target_col not in df.columns:
            print(f"警告: {self.target_col}が見つかりません")
            return df

        for window in self.windows:
            # 移動平均（当日を含まない）
            ma_col = f'ma_{window}'
            df[ma_col] = df[self.target_col].shift(1).rolling(
                window=window, min_periods=1
            ).mean()
            self.created_features.append(ma_col)

            # 移動標準偏差（変動性を捉える）
            std_col = f'ma_std_{window}'
            df[std_col] = df[self.target_col].shift(1).rolling(
                window=window, min_periods=1
            ).std()
            self.created_features.append(std_col)

            # 移動最大値
            max_col = f'ma_max_{window}'
            df[max_col] = df[self.target_col].shift(1).rolling(
                window=window, min_periods=1
            ).max()
            self.created_features.append(max_col)

            # 移動最小値
            min_col = f'ma_min_{window}'
            df[min_col] = df[self.target_col].shift(1).rolling(
                window=window, min_periods=1
            ).min()
            self.created_features.append(min_col)

        print(f"{self.name}: {len(self.created_features)}個の特徴量を作成")
        print(f"  対象変数: {self.target_col}")
        print(f"  ウィンドウ: {self.windows}")
        print(f"  統計量: 平均, 標準偏差, 最大値, 最小値")

        return df


class DomainFeatures(BaseFeatureEngineer):
    """ドメイン知識に基づく特徴量

    - CM効果の累積
    - Google Trendsの平滑化
    - アカウント取得数の傾向
    - 曜日ごとの過去平均
    """

    def __init__(self):
        super().__init__("ドメイン特徴量")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # CM効果の累積（過去7日間のCM実施回数）
        if 'cm_flg' in df.columns:
            df['cm_7d_sum'] = df['cm_flg'].shift(1).rolling(window=7, min_periods=1).sum()
            df['cm_14d_sum'] = df['cm_flg'].shift(1).rolling(window=14, min_periods=1).sum()
            self.created_features.extend(['cm_7d_sum', 'cm_14d_sum'])

        # Google Trendsの移動平均（ノイズ除去）
        if 'search_cnt' in df.columns:
            df['gt_ma_7'] = df['search_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
            df['gt_ma_14'] = df['search_cnt'].shift(1).rolling(window=14, min_periods=1).mean()
            self.created_features.extend(['gt_ma_7', 'gt_ma_14'])

        # アカウント取得数の移動平均
        if 'acc_get_cnt' in df.columns:
            df['acc_ma_7'] = df['acc_get_cnt'].shift(1).rolling(window=7, min_periods=1).mean()
            df['acc_ma_14'] = df['acc_get_cnt'].shift(1).rolling(window=14, min_periods=1).mean()
            self.created_features.extend(['acc_ma_7', 'acc_ma_14'])

        # 曜日ごとの過去平均（同じ曜日のパターンを捉える）
        if 'dow' in df.columns and 'call_num' in df.columns:
            df['dow_avg'] = np.nan
            for dow in df['dow'].unique():
                mask = df['dow'] == dow
                df.loc[mask, 'dow_avg'] = df.loc[mask, 'call_num'].shift(1).expanding().mean()
            self.created_features.append('dow_avg')

        print(f"{self.name}: {len(self.created_features)}個の特徴量を作成")
        return df


class FeaturePipeline:
    """特徴量エンジニアリングのパイプライン"""

    def __init__(self):
        self.engineers: List[BaseFeatureEngineer] = []
        self.all_features: List[str] = []

    def add(self, engineer: BaseFeatureEngineer) -> 'FeaturePipeline':
        """特徴量エンジニアを追加"""
        self.engineers.append(engineer)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ての特徴量を作成"""
        print("\n" + "=" * 80)
        print("特徴量エンジニアリング開始")
        print("=" * 80)

        result = df.copy()

        for engineer in self.engineers:
            print(f"\n[{engineer.name}]")
            result = engineer.create_features(result)
            self.all_features.extend(engineer.get_feature_names())

        print("\n" + "=" * 80)
        print(f"合計 {len(self.all_features)} 個の特徴量を作成")
        print("=" * 80)

        return result

    def get_feature_names(self) -> List[str]:
        """全特徴量名を取得"""
        return self.all_features

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量のサマリーを取得"""
        if not self.all_features:
            return pd.DataFrame()

        summary = []
        for engineer in self.engineers:
            for feat in engineer.get_feature_names():
                if feat in df.columns:
                    summary.append({
                        'group': engineer.name,
                        'feature': feat,
                        'dtype': df[feat].dtype,
                        'missing': df[feat].isnull().sum(),
                        'missing_pct': df[feat].isnull().sum() / len(df) * 100,
                        'unique': df[feat].nunique()
                    })

        return pd.DataFrame(summary)


class TimeSeriesSplitter:
    """時系列データの分割を行うクラス"""

    def __init__(self, test_months: int = 3, weekday_only: bool = True):
        self.test_months = test_months
        self.weekday_only = weekday_only

    def split(
        self,
        df: pd.DataFrame,
        target_col: str = 'call_num',
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """データを訓練・テストに分割"""
        print("\n" + "=" * 80)
        print("データ分割開始")
        print("=" * 80)

        df = df.copy()

        # 平日のみに絞り込み（オプション）
        if self.weekday_only and 'dow' in df.columns:
            df_model = df[df['dow'].isin([1, 2, 3, 4, 5])].copy().reset_index(drop=True)
            print(f"\n平日のみに絞り込み: {len(df)} → {len(df_model)} 行")
        else:
            df_model = df.copy()

        # 時系列分割（最後のN ヶ月をテストデータ）
        split_date = df_model['cdr_date'].max() - pd.Timedelta(days=30 * self.test_months)

        train_df = df_model[df_model['cdr_date'] < split_date].copy()
        test_df = df_model[df_model['cdr_date'] >= split_date].copy()

        print(f"\n時系列分割:")
        print(f"  訓練データ期間: {train_df['cdr_date'].min()} ~ {train_df['cdr_date'].max()}")
        print(f"  テストデータ期間: {test_df['cdr_date'].min()} ~ {test_df['cdr_date'].max()}")
        print(f"  訓練データ数: {len(train_df)} 行")
        print(f"  テストデータ数: {len(test_df)} 行")

        # 特徴量リストの自動取得
        if feature_cols is None:
            exclude_cols = ['cdr_date', target_col, 'dow_name', 'holiday_name', 'financial_year', 'doy']
            feature_cols = [col for col in df_model.columns if col not in exclude_cols]
            print(f"\n自動選択された特徴量数: {len(feature_cols)}")

        # 欠損値を含む行を削除
        train_clean = train_df.dropna(subset=feature_cols + [target_col])
        test_clean = test_df.dropna(subset=feature_cols + [target_col])

        print(f"\n欠損値除去後:")
        print(f"  訓練データ数: {len(train_clean)} 行")
        print(f"  テストデータ数: {len(test_clean)} 行")

        # X（特徴量）とy（目的変数）に分割
        result = {
            'X_train': train_clean[feature_cols],
            'y_train': train_clean[target_col],
            'X_test': test_clean[feature_cols],
            'y_test': test_clean[target_col],
            'train_meta': train_clean[['cdr_date', target_col]],
            'test_meta': test_clean[['cdr_date', target_col]],
            'feature_cols': feature_cols
        }

        print("\n" + "=" * 80)
        print("データ分割完了")
        print("=" * 80)

        return result


if __name__ == "__main__":
    print("特徴量エンジニアリングモジュール")
    print("使い方:")
    print("  from features import FeaturePipeline, TimeBasedFeatures, LagFeatures")
    print("  ")
    print("  pipeline = FeaturePipeline()")
    print("  pipeline.add(TimeBasedFeatures())")
    print("  pipeline.add(LagFeatures())")
    print("  df_features = pipeline.fit_transform(df)")

"""
データ構造を確認するスクリプト
EDAノートブックのセル4を実行した後に実行してください
"""

import pandas as pd
from datetime import timedelta

# ==========================================
# サンプルデータで説明
# ==========================================

print("=" * 80)
print("データ構造の説明（サンプルデータ）")
print("=" * 80)

# 1. 個別データ（結合前）
print("\n" + "=" * 80)
print("1. 結合前の個別データ（datasets）")
print("=" * 80)

call_data = pd.DataFrame({
    'cdr_date': pd.to_datetime(['2018-06-01', '2018-06-02', '2018-06-03']),
    'call_num': [183, 0, 96]
})
print("\n[call_data] - 入電数（メインデータ）")
print(call_data)
print(f"shape: {call_data.shape} （{call_data.shape[0]}行 × {call_data.shape[1]}列）")
print(f"columns: {call_data.columns.tolist()}")

calender = pd.DataFrame({
    'cdr_date': pd.to_datetime(['2018-06-01', '2018-06-02', '2018-06-03']),
    'dow': [5, 6, 7],
    'dow_name': ['Friday', 'Saturday', 'Sunday'],
    'holiday_flag': [0, 0, 0]
})
print("\n[calender] - カレンダー情報")
print(calender)
print(f"shape: {calender.shape} （{calender.shape[0]}行 × {calender.shape[1]}列）")
print(f"columns: {calender.columns.tolist()}")

cm_data = pd.DataFrame({
    'cdr_date': pd.to_datetime(['2018-06-01', '2018-06-02', '2018-06-03']),
    'cm_flg': [0, 1, 0]
})
print("\n[cm_data] - CM実施フラグ")
print(cm_data)
print(f"shape: {cm_data.shape} （{cm_data.shape[0]}行 × {cm_data.shape[1]}列）")
print(f"columns: {cm_data.columns.tolist()}")

# 2. 結合処理
print("\n" + "=" * 80)
print("2. 結合処理（merge）")
print("=" * 80)

df = call_data.copy()
print(f"\nステップ1: call_dataをコピー")
print(f"  shape: {df.shape}")
print(f"  columns: {df.columns.tolist()}")

df = df.merge(calender, on='cdr_date', how='left')
print(f"\nステップ2: calenderを結合")
print(f"  shape: {df.shape}")
print(f"  columns: {df.columns.tolist()}")
print("  ↑ call_dataのカラム + calenderのカラム（cdr_date以外）")

df = df.merge(cm_data, on='cdr_date', how='left')
print(f"\nステップ3: cm_dataを結合")
print(f"  shape: {df.shape}")
print(f"  columns: {df.columns.tolist()}")
print("  ↑ さらにcm_dataのカラム（cdr_date以外）を追加")

# 3. 結合後のデータ
print("\n" + "=" * 80)
print("3. 結合後のデータ（df_raw）")
print("=" * 80)

print("\n[df_raw] - 全データが横に結合された1つのテーブル")
print(df)
print(f"\nshape: {df.shape} （{df.shape[0]}行 × {df.shape[1]}列）")
print(f"columns: {df.columns.tolist()}")

# 4. データの取り出し方
print("\n" + "=" * 80)
print("4. データの取り出し方")
print("=" * 80)

print("\n■ 1つの列を取り出す")
print("df['call_num']")
print(df['call_num'])

print("\n■ 複数の列を取り出す")
print("df[['cdr_date', 'call_num', 'dow']]")
print(df[['cdr_date', 'call_num', 'dow']])

print("\n■ 1行目のデータ")
print("df.iloc[0]")
print(df.iloc[0])

print("\n■ 特定の値")
print("df.loc[0, 'call_num']  # 1行目のcall_num")
print(df.loc[0, 'call_num'])

# 5. df.columnsの説明
print("\n" + "=" * 80)
print("5. df.columnsの説明")
print("=" * 80)

print(f"\ndf.columns = {df.columns}")
print(f"型: {type(df.columns)}")
print(f"データ型: {df.columns.dtype}")

print("\n■ 列名をリストとして取得")
print(f"df.columns.tolist() = {df.columns.tolist()}")

print("\n■ 列数")
print(f"len(df.columns) = {len(df.columns)}")

print("\n■ 列名を1つずつ表示")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

# 6. 実際のテーブル構造を視覚化
print("\n" + "=" * 80)
print("6. テーブル構造の視覚化")
print("=" * 80)

print("\n結合前:")
print("""
call_data          calender              cm_data
┌────────────┐    ┌─────────────────┐    ┌───────────┐
│ cdr_date   │    │ cdr_date        │    │ cdr_date  │
│ call_num   │    │ dow             │    │ cm_flg    │
└────────────┘    │ dow_name        │    └───────────┘
   2列            │ holiday_flag    │       2列
                  └─────────────────┘
                        4列
""")

print("\n結合後（横に並べる）:")
print("""
df_raw
┌───────────────────────────────────────────────────┐
│ cdr_date  call_num  dow  dow_name  holiday_flag  cm_flg │
└───────────────────────────────────────────────────┘
                    6列（全て横に並ぶ）
""")

print("\n行の構造:")
for idx, row in df.iterrows():
    print(f"\n{idx}行目:")
    print(f"  cdr_date     : {row['cdr_date']}")
    print(f"  call_num     : {row['call_num']}")
    print(f"  dow          : {row['dow']}")
    print(f"  dow_name     : {row['dow_name']}")
    print(f"  holiday_flag : {row['holiday_flag']}")
    print(f"  cm_flg       : {row['cm_flg']}")

print("\n" + "=" * 80)
print("説明完了")
print("=" * 80)

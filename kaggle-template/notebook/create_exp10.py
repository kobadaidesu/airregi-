import json
import copy

# exp05を読み込み
with open(r'c:\Users\PC_User\Documents\gci_airregi\kaggle-template\notebook\exp05.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# タイトルを変更
nb['cells'][0]['source'] = ['# exp10: OOF予測値の相関分析（Best Optunaパラメータ使用）\n']

# 最適パラメータの定義（Cell 1の後に挿入）
best_params_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Best Optuna Parameters (from exp05 optimization)\n",
        "# ============================================================================\n",
        "\n",
        "BEST_PARAMS = {\n",
        "    'Ridge': {'alpha': 70.4183028501599},\n",
        "    'RandomForest': {\n",
        "        'n_estimators': 261,\n",
        "        'max_depth': 21,\n",
        "        'min_samples_split': 13,\n",
        "        'min_samples_leaf': 1,\n",
        "        'max_features': None\n",
        "    },\n",
        "    'ExtraTrees': {\n",
        "        'n_estimators': 229,\n",
        "        'max_depth': 29,\n",
        "        'min_samples_split': 16,\n",
        "        'min_samples_leaf': 1,\n",
        "        'max_features': None\n",
        "    },\n",
        "    'GradientBoosting': {\n",
        "        'n_estimators': 477,\n",
        "        'learning_rate': 0.26835579181051533,\n",
        "        'max_depth': 2,\n",
        "        'min_samples_split': 5,\n",
        "        'min_samples_leaf': 1,\n",
        "        'subsample': 0.9721678101451118\n",
        "    },\n",
        "    'HistGradientBoosting': {\n",
        "        'max_iter': 238,\n",
        "        'learning_rate': 0.015251103470998385,\n",
        "        'max_depth': 20,\n",
        "        'min_samples_leaf': 33,\n",
        "        'l2_regularization': 9.037967498117355\n",
        "    },\n",
        "    'XGBoost': {\n",
        "        'n_estimators': 4666,\n",
        "        'learning_rate': 0.18057598957444881,\n",
        "        'max_depth': 5,\n",
        "        'subsample': 0.7726782988943871,\n",
        "        'colsample_bytree': 0.6039221062901661,\n",
        "        'reg_lambda': 0.9814360532884759,\n",
        "        'reg_alpha': 1.6016986762895833\n",
        "    },\n",
        "    'LightGBM': {\n",
        "        'n_estimators': 127,\n",
        "        'learning_rate': 0.1601531217136121,\n",
        "        'num_leaves': 112,\n",
        "        'max_depth': 12,\n",
        "        'subsample': 0.9085081386743783,\n",
        "        'colsample_bytree': 0.6296178606936361,\n",
        "        'reg_lambda': 0.5211124595788266,\n",
        "        'reg_alpha': 0.5793452976256486\n",
        "    },\n",
        "    'CatBoost': {\n",
        "        'iterations': 2295,\n",
        "        'learning_rate': 0.10429705988762059,\n",
        "        'depth': 5,\n",
        "        'l2_leaf_reg': 6.359326196557493,\n",
        "        'subsample': 0.8738193035765242\n",
        "    }\n",
        "}\n",
        "\n",
        "print('Best parameters loaded from exp05 optimization')\n"
    ]
}

# Cell 1の後に挿入
nb['cells'].insert(2, best_params_cell)

# Optunaセクションを削除して、代わりにBest Paramsを使った訓練セクションに置き換える
# まず、どのセルがOptunaセクションかを特定する必要がある
# Cell 13以降がモデル訓練セクションなので、そこを探す

# 新しいモデル訓練セクション（Optunaなし、Best Params使用）
model_training_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Model Training with Best Parameters (OOF predictions saved)\n",
        "# ============================================================================\n",
        "\n",
        "from sklearn.linear_model import Ridge\n",
        "from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor\n",
        "from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor\n",
        "from xgboost import XGBRegressor\n",
        "from lightgbm import LGBMRegressor\n",
        "from catboost import CatBoostRegressor\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "\n",
        "trained_models = {}\n",
        "oof_predictions = {}  # OOF予測値を保存\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('Training models with best parameters...')\n",
        "print('='*80)\n",
        "\n",
        "# Ridge\n",
        "print('\\n[Ridge]')\n",
        "model = Ridge(**BEST_PARAMS['Ridge'], random_state=42)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['Ridge (Best)'] = model\n",
        "oof_predictions['Ridge'] = y_pred_val\n",
        "\n",
        "# RandomForest\n",
        "print('\\n[RandomForest]')\n",
        "model = RandomForestRegressor(**BEST_PARAMS['RandomForest'], random_state=42, n_jobs=-1)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['RandomForest (Best)'] = model\n",
        "oof_predictions['RandomForest'] = y_pred_val\n",
        "\n",
        "# ExtraTrees\n",
        "print('\\n[ExtraTrees]')\n",
        "model = ExtraTreesRegressor(**BEST_PARAMS['ExtraTrees'], random_state=42, n_jobs=-1)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['ExtraTrees (Best)'] = model\n",
        "oof_predictions['ExtraTrees'] = y_pred_val\n",
        "\n",
        "# GradientBoosting\n",
        "print('\\n[GradientBoosting]')\n",
        "model = GradientBoostingRegressor(**BEST_PARAMS['GradientBoosting'], random_state=42)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['GradientBoosting (Best)'] = model\n",
        "oof_predictions['GradientBoosting'] = y_pred_val\n",
        "\n",
        "# HistGradientBoosting\n",
        "print('\\n[HistGradientBoosting]')\n",
        "model = HistGradientBoostingRegressor(**BEST_PARAMS['HistGradientBoosting'], random_state=42)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['HistGradientBoosting (Best)'] = model\n",
        "oof_predictions['HistGradientBoosting'] = y_pred_val\n",
        "\n",
        "# XGBoost\n",
        "print('\\n[XGBoost]')\n",
        "model = XGBRegressor(**BEST_PARAMS['XGBoost'], random_state=42, n_jobs=-1)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['XGBoost (Best)'] = model\n",
        "oof_predictions['XGBoost'] = y_pred_val\n",
        "\n",
        "# LightGBM\n",
        "print('\\n[LightGBM]')\n",
        "model = LGBMRegressor(**BEST_PARAMS['LightGBM'], random_state=42, n_jobs=-1, verbose=-1)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['LightGBM (Best)'] = model\n",
        "oof_predictions['LightGBM'] = y_pred_val\n",
        "\n",
        "# CatBoost\n",
        "print('\\n[CatBoost]')\n",
        "model = CatBoostRegressor(**BEST_PARAMS['CatBoost'], random_state=42, verbose=0)\n",
        "model.fit(X_tr, y_tr)\n",
        "y_pred_val = model.predict(X_va)\n",
        "mae = mean_absolute_error(y_va, y_pred_val)\n",
        "print(f'Validation MAE: {mae:.4f}')\n",
        "trained_models['CatBoost (Best)'] = model\n",
        "oof_predictions['CatBoost'] = y_pred_val\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('All models trained successfully!')\n",
        "print(f'Total models: {len(trained_models)}')\n",
        "print('='*80)\n"
    ]
}

# OOF相関分析セル
oof_correlation_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# OOF Predictions Correlation Analysis\n",
        "# ============================================================================\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# OOF予測値をDataFrameに変換\n",
        "oof_df = pd.DataFrame(oof_predictions)\n",
        "\n",
        "print('\\nOOF Predictions DataFrame:')\n",
        "print(oof_df.head())\n",
        "print(f'\\nShape: {oof_df.shape}')\n",
        "\n",
        "# 相関行列を計算\n",
        "correlation_matrix = oof_df.corr()\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('OOF Predictions Correlation Matrix')\n",
        "print('='*80)\n",
        "print(correlation_matrix.round(4))\n",
        "\n",
        "# ヒートマップを作成\n",
        "plt.figure(figsize=(12, 10))\n",
        "sns.heatmap(correlation_matrix, \n",
        "            annot=True,  # 数値を表示\n",
        "            fmt='.3f',   # 小数点3桁\n",
        "            cmap='coolwarm',\n",
        "            vmin=0, vmax=1,\n",
        "            square=True,\n",
        "            linewidths=0.5,\n",
        "            cbar_kws={'label': 'Correlation'})\n",
        "plt.title('OOF Predictions Correlation Heatmap\\n(Higher correlation = Similar predictions)', \n",
        "          fontsize=14, pad=20)\n",
        "plt.xlabel('Model', fontsize=12)\n",
        "plt.ylabel('Model', fontsize=12)\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.yticks(rotation=0)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{output_dir}/oof_correlation_heatmap.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f'\\nHeatmap saved to: {output_dir}/oof_correlation_heatmap.png')\n",
        "\n",
        "# 相関行列をCSVで保存\n",
        "correlation_matrix.to_csv(f'{output_dir}/oof_correlation_matrix.csv')\n",
        "print(f'Correlation matrix saved to: {output_dir}/oof_correlation_matrix.csv')\n",
        "\n",
        "# 各モデルのMAEも一緒に表示\n",
        "print('\\n' + '='*80)\n",
        "print('Model Performance Summary')\n",
        "print('='*80)\n",
        "for model_name, pred in oof_predictions.items():\n",
        "    mae = mean_absolute_error(y_va, pred)\n",
        "    print(f'{model_name:25s} - MAE: {mae:8.4f}')\n"
    ]
}

# OOFスコアサマリーセル
oof_summary_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Save OOF Scores Summary\n",
        "# ============================================================================\n",
        "\n",
        "# OOFスコアをまとめる\n",
        "oof_scores = []\n",
        "for model_name, pred in oof_predictions.items():\n",
        "    mae = mean_absolute_error(y_va, pred)\n",
        "    oof_scores.append({\n",
        "        'Model': model_name,\n",
        "        'OOF_MAE': mae\n",
        "    })\n",
        "\n",
        "oof_scores_df = pd.DataFrame(oof_scores)\n",
        "oof_scores_df = oof_scores_df.sort_values('OOF_MAE').reset_index(drop=True)\n",
        "\n",
        "print('\\nOOF Scores Summary (sorted by MAE):')\n",
        "print(oof_scores_df.to_string(index=False))\n",
        "\n",
        "# CSV保存\n",
        "oof_scores_df.to_csv(f'{output_dir}/oof_scores_summary.csv', index=False)\n",
        "print(f'\\nOOF scores saved to: {output_dir}/oof_scores_summary.csv')\n"
    ]
}

# exp05のセル構造を確認して、Optunaセクションの代わりに新しいセルを挿入
# Cell 13からがモデルセクションと仮定

# 全てのOptunaセル（Cell 14-32あたり）を削除して、新しいセルに置き換える
# まず、どこまでがデータ準備でどこからがOptunaかを特定

# 安全のため、既存のexp05の構造を保ちつつ、Optunaセクションだけを置き換える
# Cell 0-12: 基本設定とデータ準備
# Cell 13以降: Optuna最適化とモデル訓練 → これを置き換える

# Cell 0-12を保持、Cell 13以降を新しいセルに置き換え
new_cells = nb['cells'][:13]  # 基本設定とデータ準備部分を保持

# 新しいモデル訓練セクションを追加
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["---\n", "## Model Training with Best Parameters\n"]
})

new_cells.append(model_training_cell)
new_cells.append(oof_correlation_cell)
new_cells.append(oof_summary_cell)

# 予測と保存セクション（元のexp05から流用）
# テストデータ予測セル
test_prediction_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Test Predictions\n",
        "# ============================================================================\n",
        "\n",
        "test_predictions = {}\n",
        "\n",
        "for model_name, model in trained_models.items():\n",
        "    pred = model.predict(test_df[feature_cols])\n",
        "    test_predictions[model_name] = pred\n",
        "    print(f'{model_name}: {len(pred)} predictions made')\n",
        "\n",
        "print(f'\\nTotal models: {len(test_predictions)}')\n"
    ]
}

# 提出ファイル作成セル
submission_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Create Submission Files\n",
        "# ============================================================================\n",
        "\n",
        "import os\n",
        "\n",
        "for model_name, pred in test_predictions.items():\n",
        "    submission = pd.DataFrame({\n",
        "        'id': test_df['id'],\n",
        "        'target_next_day': pred\n",
        "    })\n",
        "    \n",
        "    # モデル名をファイル名に使用（スペースをアンダースコアに）\n",
        "    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')\n",
        "    filepath = f'{output_dir}/submission_{safe_name}.csv'\n",
        "    \n",
        "    submission.to_csv(filepath, index=False)\n",
        "    print(f'Saved: {filepath}')\n",
        "\n",
        "print('\\nAll submission files created!')\n"
    ]
}

new_cells.append(test_prediction_cell)
new_cells.append(submission_cell)

# 最後にSHAPセクションを追加（exp05から）
# SHAPセクションはexp05のCell 33-34あたりにある
# ここでは簡略化して、主要モデルのSHAPのみ追加

shap_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# SHAP Analysis (for key models)\n",
        "# ============================================================================\n",
        "\n",
        "import shap\n",
        "\n",
        "# サンプル数を制限\n",
        "sample_size = min(100, len(X_va))\n",
        "X_explain = X_va.sample(n=sample_size, random_state=42)\n",
        "\n",
        "print(f'SHAP analysis on {sample_size} samples...')\n",
        "\n",
        "# XGBoost SHAP\n",
        "if 'XGBoost (Best)' in trained_models:\n",
        "    print('\\n[XGBoost SHAP]')\n",
        "    model = trained_models['XGBoost (Best)']\n",
        "    explainer = shap.TreeExplainer(model)\n",
        "    shap_values = explainer.shap_values(X_explain)\n",
        "    \n",
        "    plt.figure(figsize=(12, 8))\n",
        "    shap.summary_plot(shap_values, X_explain, show=False)\n",
        "    plt.title('XGBoost - SHAP Summary Plot')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(f'{output_dir}/shap_xgboost_summary.png', dpi=300, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "# CatBoost SHAP\n",
        "if 'CatBoost (Best)' in trained_models:\n",
        "    print('\\n[CatBoost SHAP]')\n",
        "    model = trained_models['CatBoost (Best)']\n",
        "    explainer = shap.TreeExplainer(model)\n",
        "    shap_values = explainer.shap_values(X_explain)\n",
        "    \n",
        "    plt.figure(figsize=(12, 8))\n",
        "    shap.summary_plot(shap_values, X_explain, show=False)\n",
        "    plt.title('CatBoost - SHAP Summary Plot')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(f'{output_dir}/shap_catboost_summary.png', dpi=300, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "print('\\nSHAP analysis completed!')\n"
    ]
}

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["---\n", "## SHAP Analysis\n"]
})
new_cells.append(shap_cell)

# 完了メッセージ
final_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Summary\n",
        "\n",
        "**exp10: OOF Predictions Correlation Analysis**\n",
        "\n",
        "- ベースライン: exp05\n",
        "- Best Optunaパラメータを使用\n",
        "- 各モデルのOOF予測値の相関を分析\n",
        "- 相関ヒートマップ（数値付き）を出力\n",
        "\n",
        "### Key Outputs:\n",
        "1. `oof_correlation_heatmap.png` - OOF予測値の相関ヒートマップ\n",
        "2. `oof_correlation_matrix.csv` - 相関行列（CSV）\n",
        "3. `oof_scores_summary.csv` - 各モデルのOOF MAE\n",
        "4. Submission files for all models\n"
    ]
}

new_cells.append(final_cell)

# ノートブックを更新
nb['cells'] = new_cells

# exp10として保存
with open(r'c:\Users\PC_User\Documents\gci_airregi\kaggle-template\notebook\exp10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('exp10.ipynb created successfully!')
print('\nKey changes from exp05:')
print('1. Using best Optuna parameters (no optimization)')
print('2. Saving OOF predictions for all models')
print('3. Computing correlation matrix of OOF predictions')
print('4. Generating heatmap with correlation values')
print('5. Saving correlation analysis to CSV')

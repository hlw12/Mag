# 震级预测实验报告

## 实验信息
- **实验名称**: magnitude_prediction_experiment_500
- **开始时间**: 2025-09-14T01:59:48.518975
- **完成时间**: 2025-09-14T02:11:58.815213
- **状态**: completed

## 模型信息

- **模型参数数量**: 6,337,311
- **模型大小**: 24.2 MB
- **模型特性**: ['Multi-scale temporal convolution', 'Adaptive spectral attention', 'Cross-modal fusion', 'Magnitude-aware pooling', 'Transformer encoding']

## 训练信息
- **训练轮数**: 231
- **最佳验证损失**: 0.1626

## Final Evaluation 结果

### 回归指标
- **MAE**: 0.3304
- **RMSE**: 0.4492
- **R²**: 0.8283

### 质量等级分析
- **优秀预测率 (≤0.2)**: 42.1%
- **良好预测率 (≤0.3)**: 59.7%
- **可接受预测率 (≤0.5)**: 77.8%

## 实验文件
- **config**: `experiments\magnitude_prediction_experiment_500\configs\experiment_config.json`
- **training_history**: `experiments\magnitude_prediction_experiment_500\logs\training_history.json`
- **model**: `experiments\magnitude_prediction_experiment_500\models\best_model_20250914_021154.pth`
- **final_evaluation_results**: `experiments\magnitude_prediction_experiment_500\results\final_evaluation_20250914_021154.json`
- **plots**: `{'training_history': 'experiments\\magnitude_prediction_experiment_500\\plots\\training_history_20250914_021154.png', 'evaluation_results': 'experiments\\magnitude_prediction_experiment_500\\plots\\evaluation_results_20250914_021154.png', 'prediction_analysis': 'experiments\\magnitude_prediction_experiment_500\\plots\\prediction_analysis_20250914_021154.png', 'association_analysis': None}`

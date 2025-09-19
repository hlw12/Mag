#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 16:45
# @Author  : 上头欢乐送、
# @File    : Trainer.py
# @Software: PyCharm
# 学习新思想，争做新青年
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.nn import MSELoss
from tqdm import tqdm

class R2Loss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true, y_pred: shape [B]
        y_true_mean = torch.mean(y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        return ss_res / (ss_tot + self.eps)


class MagnitudeTrainer:
    """震级预测模型训练器"""

    def __init__(self, model,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            (wave, spec), target = batch
            wave = wave.to(self.device)  # 时域波形 [B, 3, T]
            spec = spec.to(self.device)  # 频谱图     [B, 3, F, L]
            target = target.to(self.device)  # 震级标签   [B,]

            optimizer.zero_grad()
            output = self.model((wave, spec))
            temp_loss = nn.SmoothL1Loss()
            loss = criterion(output, target)+ 0.1 * temp_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                (wave, spec), target = batch
                wave = wave.to(self.device)
                spec = spec.to(self.device)
                target = target.to(self.device)
                output = self.model((wave, spec))
                loss = criterion(output, target)

                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)

        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        return avg_loss, mae, mse, rmse, r2, predictions, targets

    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001,
              patience=15, save_path='best_magnitude_model.pth'):
        """完整训练流程"""
        criterion = MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience // 3, factor=0.5, verbose=True
        )

        patience_counter = 0

        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            val_loss, mae, mse, rmse, r2, _, _ = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }, save_path)
                print(f"保存最佳模型: {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在epoch {epoch + 1}")
                break

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def evaluate_model(model, test_loader, device, association_threshold=0.3):
    model.eval()
    predictions = []
    targets = []
    sample_indices = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Enhanced Testing"):
            (wave, spec), target = batch
            wave = wave.to(device)
            spec = spec.to(device)
            target = target.to(device)
            output = model((wave, spec))

            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    print(f"\n{'=' * 60}")
    print(f"Evaluation results:")
    print(f"{'=' * 60}")

    print(f"\nBasic regression index:")
    print(f" MAE: {mae:.4f}")
    print(f" MSE: {mse:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" R²: {r2:.4f}")

    association_results = association_evaluation(predictions, targets)
    # create_visualization(results=association_results, predictions=predictions, targets=targets)

    return {
        'regression_metrics': {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2},
        'association_results': association_results,
        'predictions': predictions,
        'targets': targets
    }


def create_visualization(results, predictions, targets):
    errors = np.abs(predictions - targets)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    quality_metrics = results['quality_metrics']

    excellent_count = quality_metrics['excellent']['count']
    good_count = quality_metrics['good']['count'] - excellent_count
    acceptable_count = quality_metrics['acceptable']['count'] - quality_metrics['good']['count']
    poor_count = len(predictions) - quality_metrics['acceptable']['count']

    counts = [excellent_count, good_count, acceptable_count, poor_count]
    labels = ['优秀\n(≤0.2)', '良好\n(0.2-0.3)', '可接受\n(0.3-0.5)', '需改进\n(>0.5)']
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    non_zero_mask = np.array(counts) > 0
    filtered_counts = np.array(counts)[non_zero_mask]
    filtered_labels = np.array(labels)[non_zero_mask]
    filtered_colors = np.array(colors)[non_zero_mask]

    wedges, texts, autotexts = ax.pie(filtered_counts, labels=filtered_labels,
                                     colors=filtered_colors, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 10})
    ax.set_title('预测质量分布', fontsize=12, fontweight='bold')
    ax = axes[0, 1]
    cm = results['classification_metrics']['confusion_matrix']
    magnitude_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    bin_labels = [f"{magnitude_bins[i]:.1f}-{magnitude_bins[i+1]:.1f}"
                  for i in range(len(magnitude_bins)-1)]

    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=bin_labels, yticklabels=bin_labels,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('震级区间混淆矩阵', fontsize=12, fontweight='bold')
    ax.set_xlabel('预测震级区间')
    ax.set_ylabel('真实震级区间')
    ax = axes[0, 2]
    scatter = ax.scatter(targets, predictions, c=errors, cmap='viridis',
                        alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)

    ax.set_xlabel('真实震级')
    ax.set_ylabel('预测震级')
    ax.set_title('预测vs真实值\n(颜色=预测误差)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('预测误差', rotation=270, labelpad=15)
    ax = axes[1, 0]
    levels = ['优秀', '良好', '可接受']
    coverage_rates = [quality_metrics['excellent']['recall'],
                     quality_metrics['good']['recall'],
                     quality_metrics['acceptable']['recall']]
    f1_scores = [quality_metrics['excellent']['f1_score'],
                quality_metrics['good']['f1_score'],
                quality_metrics['acceptable']['f1_score']]

    x = np.arange(len(levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, coverage_rates, width, label='覆盖率',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1分数',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('质量等级')
    ax.set_ylabel('指标值')
    ax.set_title('质量等级性能指标', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 1]
    n, bins, patches = ax.hist(errors, bins=30, alpha=0.7, color='skyblue',
                              edgecolor='black', linewidth=0.5)

    ax.axvline(x=0.2, color='green', linestyle='--', linewidth=2,
              label='优秀阈值(0.2)', alpha=0.8)
    ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=2,
              label='良好阈值(0.3)', alpha=0.8)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
              label='可接受阈值(0.5)', alpha=0.8)

    mean_error = errors.mean()
    ax.axvline(x=mean_error, color='purple', linestyle='-', linewidth=2,
              label=f'平均误差({mean_error:.3f})', alpha=0.8)

    ax.set_xlabel('预测误差')
    ax.set_ylabel('频次')
    ax.set_title('预测误差分布', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1, 2]

    class_report = results['classification_metrics']['class_report']
    bin_labels = [f"{magnitude_bins[i]:.1f}-{magnitude_bins[i+1]:.1f}"
                  for i in range(len(magnitude_bins)-1)]

    f1_values = []
    for label in bin_labels:
        f1_values.append(class_report[label]['f1-score'])

    angles = np.linspace(0, 2*np.pi, len(bin_labels), endpoint=False).tolist()
    f1_values += f1_values[:1]  # 闭合
    angles += angles[:1]

    ax.plot(angles, f1_values, 'o-', linewidth=2, color='#e74c3c', markersize=6)
    ax.fill(angles, f1_values, alpha=0.25, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('各震级区间F1性能', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    for angle, value in zip(angles[:-1], f1_values[:-1]):
        ax.text(angle, value + 0.05, f'{value:.2f}',
               ha='center', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return fig


def association_evaluation(predictions, targets):
    """
    改进的关联关系评估，使用更合理的关联定义
    """
    errors = np.abs(predictions - targets)
    n_samples = len(predictions)

    results = {}

    print("Quality level correlation assessment: 0.2/0.3/0.5")
    excellent_mask = errors <= 0.2
    good_mask = errors <= 0.3
    acceptable_mask = errors <= 0.5
    excellent_samples = set(np.where(excellent_mask)[0])
    good_samples = set(np.where(good_mask)[0])
    acceptable_samples = set(np.where(acceptable_mask)[0])
    target_samples = set(range(n_samples))
    quality_metrics = {}
    for level, pred_set in [
        ('excellent', excellent_samples),
        ('good', good_samples),
        ('acceptable', acceptable_samples)
    ]:
        tp = len(pred_set)
        fn = len(target_samples - pred_set)

        precision = 1.0
        recall = tp / len(target_samples) if len(target_samples) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        quality_metrics[level] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': tp / n_samples,
            'count': tp
        }

        print(f"  {level.capitalize():12}: Coverage={recall:.3f}, F1={f1:.3f}, 样本数={tp}")

    results['quality_metrics'] = quality_metrics

    print("Magnitude range classification assessment:")
    magnitude_bins = [0.5,1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    bin_labels = [f"{magnitude_bins[i]:.1f}-{magnitude_bins[i + 1]:.1f}"
                  for i in range(len(magnitude_bins) - 1)]

    true_labels = np.digitize(targets, magnitude_bins) - 1
    pred_labels = np.digitize(predictions, magnitude_bins) - 1
    true_labels = np.clip(true_labels, 0, len(bin_labels) - 1)
    pred_labels = np.clip(pred_labels, 0, len(bin_labels) - 1)

    class_report = classification_report(
        true_labels, pred_labels,
        target_names=bin_labels,
        output_dict=True,
        zero_division=0
    )

    print(" Magnitude range classification accuracy:")
    for i, label in enumerate(bin_labels):
        metrics = class_report[label]
        count = (true_labels == i).sum()
        print(f"    {label}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, n={count}")

    overall_accuracy = (true_labels == pred_labels).mean()
    macro_f1 = class_report['macro avg']['f1-score']
    weighted_f1 = class_report['weighted avg']['f1-score']

    print(f" Overall classification accuracy: {overall_accuracy:.3f}")
    print(f" Macro average F1: {macro_f1:.3f}")
    print(f" Weighted average F1: {weighted_f1:.3f}")

    results['classification_metrics'] = {
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_report': class_report,
        'confusion_matrix': confusion_matrix(true_labels, pred_labels)
    }
    return results
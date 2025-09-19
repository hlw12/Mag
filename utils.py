#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 21:15
# @Author  : 上头欢乐送、
# @File    : utils.py
# @Software: PyCharm
# 学习新思想，争做新青年

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 17:45
# @Author  : 上头欢乐送、
# @File    : magnet_utils.py
# @Software: PyCharm
# 学习新思想，争做新青年

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class MagnitudeLoss(nn.Module):
    """
    自定义震级损失函数
    结合了MSE和MAE，并考虑震级的物理意义
    """

    def __init__(self, alpha=0.7, beta=0.3, magnitude_weight=True):
        super(MagnitudeLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta  # MAE权重
        self.magnitude_weight = magnitude_weight
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)

        combined_loss = self.alpha * mse_loss + self.beta * mae_loss
        if self.magnitude_weight:
            weights = torch.exp((targets - 3.0) / 2.0)  # 以3.0为基准
            weighted_loss = combined_loss * weights
            return weighted_loss.mean()
        else:
            return combined_loss.mean()


class FocalMagnitudeLoss(nn.Module):
    """
    Focal Loss变体，用于震级估计
    对难以预测的样本给予更多关注
    """

    def __init__(self, gamma=2.0, alpha=1.0):
        super(FocalMagnitudeLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):
        error = torch.abs(predictions - targets)
        focal_weight = torch.pow(error / (error.max() + 1e-7), self.gamma)
        loss = self.alpha * focal_weight * torch.pow(error, 2)
        return loss.mean()


class WarmupCosineScheduler(_LRScheduler):
    """
    带预热的余弦退火学习率调度器
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]


def plot_magnitude_confusion_matrix(predictions, targets, save_path=None):
    """
    绘制震级预测的混淆矩阵
    """
    bins = np.arange(2.0, 7.5, 0.5)
    pred_bins = np.digitize(predictions, bins)
    true_bins = np.digitize(targets, bins)
    cm = confusion_matrix(true_bins, pred_bins)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'{b:.1f}' for b in bins[:-1]],
                yticklabels=[f'{b:.1f}' for b in bins[:-1]])
    plt.xlabel('Predicted Magnitude')
    plt.ylabel('True Magnitude')
    plt.title('Magnitude Prediction Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_waveform_features(waveform_batch, model, device):
    """
    分析波形特征对震级预测的影响
    使用梯度分析找出重要的时间段
    """
    model.eval()
    waveform_batch = waveform_batch.to(device).requires_grad_(True)
    predictions = model(waveform_batch)
    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=waveform_batch,
        create_graph=False
    )[0]

    # 计算特征重要性（按时间步的平均梯度）
    feature_importance = gradients.abs().mean(dim=(0, 2)).cpu().numpy()

    return feature_importance


def create_augmented_batch(batch, augmentation_params):
    """
    数据增强：为地震波形创建增强版本

    Args:
        batch: 原始波形批次 (batch_size, seq_len, channels)
        augmentation_params: 增强参数字典

    Returns:
        增强后的批次
    """
    augmented_batch = batch.clone()
    if 'noise_level' in augmentation_params:
        noise = torch.randn_like(batch) * augmentation_params['noise_level']
        augmented_batch += noise
    if 'time_shift' in augmentation_params:
        shift = augmentation_params['time_shift']
        if shift > 0:
            augmented_batch = F.pad(augmented_batch[:, shift:, :], (0, 0, 0, shift))
        elif shift < 0:
            augmented_batch = F.pad(augmented_batch[:, :shift, :], (0, 0, -shift, 0))

    # 3. 振幅缩放
    if 'amplitude_scale' in augmentation_params:
        scale = augmentation_params['amplitude_scale']
        augmented_batch *= scale

    # 4. 通道置换
    if 'channel_shuffle' in augmentation_params and augmentation_params['channel_shuffle']:
        perm = torch.randperm(3)
        augmented_batch = augmented_batch[:, :, perm]

    return augmented_batch


class EarlyStopping:
    """
    早停机制的改进版本
    """

    def __init__(self, patience=10, min_delta=0.0001, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: 性能提升，最佳分数: {self.best_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: 没有提升 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: 触发早停！")

        return self.early_stop


def save_model_with_metadata(model, optimizer, epoch, metrics, save_path):
    """
    保存模型及其元数据
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': {
            'input_channels': 3,
            'lstm_hidden_size': getattr(model, 'lstm_hidden_size', 100),
            'dropout_rate': getattr(model, 'dropout_rate', 0.3),
        },
        'timestamp': torch.tensor(np.datetime64('now', 's').astype(int))
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")


def compute_gutenberg_richter_statistics(magnitudes, predictions):
    """
    计算Gutenberg-Richter关系的统计信息
    这对于验证模型预测的物理合理性很重要
    """
    # 计算不同震级阈值下的累积频率
    mag_thresholds = np.arange(2.0, 7.0, 0.2)
    true_counts = []
    pred_counts = []

    for threshold in mag_thresholds:
        true_counts.append(np.sum(magnitudes >= threshold))
        pred_counts.append(np.sum(predictions >= threshold))

    # 对数变换（G-R关系是对数线性的）
    log_true_counts = np.log10(np.array(true_counts) + 1)
    log_pred_counts = np.log10(np.array(pred_counts) + 1)

    # 计算b值（G-R关系的斜率）
    from scipy import stats
    true_slope, _, _, _, _ = stats.linregress(mag_thresholds, log_true_counts)
    pred_slope, _, _, _, _ = stats.linregress(mag_thresholds, log_pred_counts)

    return {
        'mag_thresholds': mag_thresholds,
        'true_counts': true_counts,
        'pred_counts': pred_counts,
        'true_b_value': -true_slope,
        'pred_b_value': -pred_slope,
        'b_value_error': abs(true_slope - pred_slope)
    }


def plot_training_diagnostics(trainer, save_dir):
    """
    绘制详细的训练诊断图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 损失曲线
    ax = axes[0, 0]
    epochs = range(1, len(trainer.train_losses) + 1)
    ax.plot(epochs, trainer.train_losses, 'b-', label='Train Loss')
    ax.plot(epochs, trainer.val_losses, 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 学习率变化
    ax = axes[0, 1]
    if hasattr(trainer, 'lr_history'):
        ax.plot(epochs, trainer.lr_history, 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # 3. 验证集性能指标
    ax = axes[1, 0]
    if hasattr(trainer, 'val_mae_history'):
        ax.plot(epochs, trainer.val_mae_history, 'b-', label='MAE')
        ax.plot(epochs, trainer.val_rmse_history, 'r-', label='RMSE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.set_title('Validation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. 梯度范数
    ax = axes[1, 1]
    if hasattr(trainer, 'grad_norm_history'):
        ax.plot(epochs, trainer.grad_norm_history, 'purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.close()


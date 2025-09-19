#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 15:48
# @Author  : 上头欢乐送、
# @File    : saver.py
# @Software: PyCharm
# 学习新思想，争做新青年

import json
import pickle
import seaborn as sns
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


class ExperimentLogger:
    """
    完整的实验记录系统
    保存模型、训练过程、评估结果、可视化图表等所有实验内容
    """

    def __init__(self, base_dir="experiments", experiment_name=None):
        self.base_dir = Path(base_dir)

        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"magnitude_exp_{timestamp}"
        else:
            self.experiment_name = experiment_name

        self.exp_dir = self.base_dir / self.experiment_name
        self.create_directory_structure()
        self.experiment_log = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.datetime.now().isoformat(),
            'status': 'running',
            'config': {},
            'training_history': {},
            'evaluation_results': {},
            'model_info': {},
            'files': {}
        }

        print(f"Experiment started: {self.experiment_name}")
        print(f"Experiment directory: {self.exp_dir}")

    def create_directory_structure(self):
        directories = [
            'models',  # 保存训练好的模型
            'checkpoints',  # 保存训练检查点
            'logs',  # 保存训练日志
            'plots',  # 保存可视化图表
            'data',  # 保存实验数据
            'configs',  # 保存配置文件
            'results',  # 保存评估结果
            'reports'  # 保存实验报告
        ]
        for dir_name in directories:
            (self.exp_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def save_config(self, config_dict, config_name="experiment_config"):
        config_path = self.exp_dir / 'configs' / f'{config_name}.json'
        serializable_config = self._make_serializable(config_dict)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        self.experiment_log['config'] = serializable_config
        self.experiment_log['files']['config'] = str(config_path)

        print(f"Configuration saved: {config_path}")

    def save_training_history(self, trainer, epoch=None):
        """保存训练历史"""
        training_data = {
            'train_losses': getattr(trainer, 'train_losses', []),
            'val_losses': getattr(trainer, 'val_losses', []),
            'best_val_loss': getattr(trainer, 'best_val_loss', None),
            'current_epoch': epoch,
            'total_epochs': len(getattr(trainer, 'train_losses', [])),
            'lr_history': getattr(trainer, 'lr_history', []),
            'grad_norm_history': getattr(trainer, 'grad_norm_history', [])
        }

        history_path = self.exp_dir / 'logs' / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        pickle_path = self.exp_dir / 'logs' / 'training_history.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(training_data, f)

        self.experiment_log['training_history'] = training_data
        self.experiment_log['files']['training_history'] = str(history_path)
        print(f"Training history saved: {history_path}")

    def save_model(self, model, optimizer=None, epoch=None, metrics=None, model_name="best_model"):
        """保存模型和相关信息"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.exp_dir / 'models' / f'{model_name}_{timestamp}.pth'
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'save_time': datetime.datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics or {}
        }

        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if hasattr(model, 'get_model_info'):
            save_dict['model_info'] = model.get_model_info()
        else:
            total_params = sum(p.numel() for p in model.parameters())
            save_dict['model_info'] = {
                'total_parameters': total_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)
            }
        self.experiment_log['model_info'] = save_dict['model_info']
        torch.save(save_dict, model_path)
        latest_path = self.exp_dir / 'models' / 'latest_model.pth'
        try:
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(model_path.name)
        except (OSError, NotImplementedError):
            import shutil
            if latest_path.exists():
                latest_path.unlink()
            shutil.copy2(model_path, latest_path)
        self.experiment_log['files']['model'] = str(model_path)

        print(f"Model saved: {model_path}")

        return model_path

    def save_evaluation_results(self, results, eval_name="final_evaluation"):
        """保存评估结果"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.exp_dir / 'results' / f'{eval_name}_{timestamp}.json'
        serializable_results = self._make_serializable(results)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        if 'predictions' in results and 'targets' in results:
            data_path = self.exp_dir / 'data' / f'{eval_name}_predictions_{timestamp}.npz'
            np.savez(data_path,
                     predictions=results['predictions'],
                     targets=results['targets'])
            serializable_results['raw_data_path'] = str(data_path)

        self.experiment_log['evaluation_results'][eval_name] = serializable_results
        self.experiment_log['files'][f'{eval_name}_results'] = str(results_path)

        print(f"Evaluation results saved: {results_path}")

        return results_path

    def save_plots(self, trainer=None, results=None, predictions=None, targets=None):
        """保存所有相关图表"""
        print(results.keys())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_paths = {}
        if trainer is not None and hasattr(trainer, 'train_losses'):
            plot_path = self._save_training_plots(trainer, timestamp)
            plot_paths['training_history'] = plot_path

        if results is not None:
            plot_path = self._save_evaluation_plots(results, timestamp)
            plot_paths['evaluation_results'] = plot_path

        if predictions is not None and targets is not None:
            plot_path = self._save_prediction_plots(predictions, targets, timestamp)
            plot_paths['prediction_analysis'] = plot_path

        if results is not None and 'association_results' in results:
            plot_path = self._save_association_plots(results['association_results'], timestamp)
            plot_paths['association_analysis'] = plot_path

        self.experiment_log['files']['plots'] = plot_paths
        print(f"Chart saved to: {self.exp_dir / 'plots'}")

        return plot_paths

    def _save_training_plots(self, trainer, timestamp):
        """保存训练过程图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax = axes[0, 0]
        epochs = range(1, len(trainer.train_losses) + 1)
        ax.plot(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.semilogy(epochs, trainer.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.semilogy(epochs, trainer.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training History (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        if hasattr(trainer, 'lr_history') and trainer.lr_history:
            ax.plot(epochs, trainer.lr_history, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR History Available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate Schedule')

        ax = axes[1, 1]
        if hasattr(trainer, 'grad_norm_history') and trainer.grad_norm_history:
            ax.plot(epochs, trainer.grad_norm_history, 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm History')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Gradient Norm History',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Gradient Norm History')

        plt.tight_layout()

        plot_path = self.exp_dir / 'plots' / f'training_history_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _save_evaluation_plots(self, results, timestamp):
        if 'predictions' not in results or 'targets' not in results:
            print("Missing prediction data, skipping evaluation chart save!")
            return None

        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        errors = np.abs(predictions - targets)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        individual_plots_dir = self.exp_dir / 'plots' / 'individual'
        individual_plots_dir.mkdir(exist_ok=True)
        individual_plot_paths = []

        # ===== 子图1: 预测vs真实值散点图 (增强版) =====
        ax = axes[0, 0]
        scatter = ax.scatter(targets, predictions, c=errors, cmap='viridis',
                             alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

        min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8,
                label='Perfect Prediction')
        x_range = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_range, x_range - 0.2, x_range + 0.2,
                        alpha=0.15, color='green', label='Excellent Zone (±0.2)')
        ax.plot(x_range, x_range + 0.2, '--', color='green', linewidth=2, alpha=0.8)
        ax.plot(x_range, x_range - 0.2, '--', color='green', linewidth=2, alpha=0.8)
        ax.plot(x_range, x_range + 0.3, '--', color='orange', linewidth=2, alpha=0.8,
                label='Good Boundary (±0.3)')
        ax.plot(x_range, x_range - 0.3, '--', color='orange', linewidth=2, alpha=0.8)
        ax.plot(x_range, x_range + 0.5, '--', color='red', linewidth=2, alpha=0.8,
                label='Acceptable Boundary (±0.5)')
        ax.plot(x_range, x_range - 0.5, '--', color='red', linewidth=2, alpha=0.8)

        ax.set_xlabel('True Magnitude')
        ax.set_ylabel('Predicted Magnitude')
        ax.set_title('Predictions vs True Values (with Quality Zones)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # 设置相等的坐标轴比例
        ax.set_aspect('equal', adjustable='box')

        cbar = plt.colorbar(scatter, ax=ax, label='Prediction Error', shrink=0.8)

        # 单独保存子图1
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)

        scatter1 = ax1.scatter(targets, predictions, c=errors, cmap='viridis',
                               alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8,
                 label='Perfect Prediction')
        ax1.fill_between(x_range, x_range - 0.2, x_range + 0.2,
                         alpha=0.15, color='green', label='Excellent Zone (±0.2)')
        ax1.plot(x_range, x_range + 0.2, '--', color='green', linewidth=2, alpha=0.8)
        ax1.plot(x_range, x_range - 0.2, '--', color='green', linewidth=2, alpha=0.8)
        ax1.plot(x_range, x_range + 0.3, '--', color='orange', linewidth=2, alpha=0.8,
                 label='Good Boundary (±0.3)')
        ax1.plot(x_range, x_range - 0.3, '--', color='orange', linewidth=2, alpha=0.8)
        ax1.plot(x_range, x_range + 0.5, '--', color='red', linewidth=2, alpha=0.8,
                 label='Acceptable Boundary (±0.5)')
        ax1.plot(x_range, x_range - 0.5, '--', color='red', linewidth=2, alpha=0.8)

        ax1.set_xlabel('True Magnitude', fontsize=12)
        ax1.set_ylabel('Predicted Magnitude', fontsize=12)
        ax1.set_title('Predictions vs True Values (with Quality Zones)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_aspect('equal', adjustable='box')

        plt.colorbar(scatter1, ax=ax1, label='Prediction Error')

        plot1_path = individual_plots_dir / f'01_predictions_vs_targets_enhanced_{timestamp}.png'
        fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        individual_plot_paths.append(str(plot1_path))

        # ===== 子图2: 混淆矩阵 =====
        ax = axes[0, 1]
        if 'association_results' in results and 'confusion_matrix' in results['association_results']['classification_metrics']:
            magnitude_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
            bin_labels = [f"{magnitude_bins[i]:.1f}-{magnitude_bins[i + 1]:.1f}"
                          for i in range(len(magnitude_bins) - 1)]
            cm = results['association_results']['classification_metrics']['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=bin_labels, yticklabels=bin_labels,
                        ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Magnitude Range Confusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Magnitude Range', fontsize=10)
            ax.set_ylabel('True Magnitude Range', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

            # 单独保存混淆矩阵
            fig2 = plt.figure(figsize=(12, 10))
            ax2 = fig2.add_subplot(111)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=bin_labels, yticklabels=bin_labels,
                        ax=ax2, cbar_kws={'shrink': 0.8})
            ax2.set_title('Magnitude Range Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
            ax2.set_xlabel('Predicted Magnitude Range', fontsize=14)
            ax2.set_ylabel('True Magnitude Range', fontsize=14)
            ax2.tick_params(axis='x', rotation=45, labelsize=12)
            ax2.tick_params(axis='y', rotation=0, labelsize=12)

            # 添加准确率信息
            total_samples = cm.sum()
            correct_predictions = np.diag(cm).sum()
            accuracy = correct_predictions / total_samples
            # ax2.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.3f}\nTotal Samples: {total_samples}',
            #          transform=ax2.transAxes, fontsize=12, verticalalignment='top',
            #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plot2_path = individual_plots_dir / f'02_confusion_matrix_{timestamp}.png'
            fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            individual_plot_paths.append(str(plot2_path))
        else:
            ax.text(0.5, 0.5, 'No Confusion Matrix Data Available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Confusion Matrix (No Data)')

        # ===== 子图3: 误差分布直方图 =====
        ax = axes[0, 2]
        n, bins, patches = ax.hist(errors, bins=30, alpha=0.7, color='skyblue',
                                   edgecolor='black', linewidth=0.5)
        ax.axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='Excellent (0.2)')
        ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Good (0.3)')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Acceptable (0.5)')
        ax.axvline(x=errors.mean(), color='purple', linestyle='-', linewidth=2,
                   label=f'Mean ({errors.mean():.3f})')
        ax.set_xlabel('Prediction Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Error Distribution', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 单独保存子图3
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        n3, bins3, patches3 = ax3.hist(errors, bins=50, alpha=0.7, color='skyblue',
                                       edgecolor='black', linewidth=0.5, density=True)
        ax3.axvline(x=0.2, color='green', linestyle='--', linewidth=3, label='Excellent (0.2)')
        ax3.axvline(x=0.3, color='orange', linestyle='--', linewidth=3, label='Good (0.3)')
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=3, label='Acceptable (0.5)')
        ax3.axvline(x=errors.mean(), color='purple', linestyle='-', linewidth=2,
                    label=f'Mean ({errors.mean():.3f})')
        ax3.axvline(x=np.median(errors), color='brown', linestyle='-', linewidth=2,
                    label=f'Median ({np.median(errors):.3f})')
        ax3.set_xlabel('Prediction Error', fontsize=12)
        ax3.set_ylabel('Probability Density', fontsize=12)
        ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f'Statistics:\nStd Dev: {errors.std():.3f}\nMax Error: {errors.max():.3f}\nSamples: {len(errors)}'
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plot3_path = individual_plots_dir / f'03_error_distribution_{timestamp}.png'
        fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        individual_plot_paths.append(str(plot3_path))

        # ===== 子图4: 质量分布饼图 =====
        ax = axes[1, 0]
        excellent_count = (errors <= 0.2).sum()
        good_count = ((errors > 0.2) & (errors <= 0.3)).sum()
        acceptable_count = ((errors > 0.3) & (errors <= 0.5)).sum()
        poor_count = (errors > 0.5).sum()

        counts = [excellent_count, good_count, acceptable_count, poor_count]
        labels = ['Excellent\n(≤0.2)', 'Good\n(0.2-0.3)', 'Acceptable\n(0.3-0.5)', 'Poor\n(>0.5)']
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

        non_zero_mask = np.array(counts) > 0
        if non_zero_mask.any():
            filtered_counts = np.array(counts)[non_zero_mask]
            filtered_labels = np.array(labels)[non_zero_mask]
            filtered_colors = np.array(colors)[non_zero_mask]

            wedges, texts, autotexts = ax.pie(filtered_counts, labels=filtered_labels,
                                              colors=filtered_colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 9})
        ax.set_title('Prediction Quality Distribution', fontsize=12)

        # 单独保存子图4
        fig4 = plt.figure(figsize=(8, 8))
        ax4 = fig4.add_subplot(111)
        if non_zero_mask.any():
            wedges4, texts4, autotexts4 = ax4.pie(filtered_counts, labels=filtered_labels,
                                                  colors=filtered_colors, autopct='%1.1f%%',
                                                  startangle=90, textprops={'fontsize': 12})
            ax4.legend(wedges4, [f'{label}: {count}' for label, count in zip(filtered_labels, filtered_counts)],
                       title="Sample Statistics", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax4.set_title('Prediction Quality Distribution', fontsize=14, fontweight='bold', pad=20)

        plot4_path = individual_plots_dir / f'04_quality_distribution_{timestamp}.png'
        fig4.savefig(plot4_path, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        individual_plot_paths.append(str(plot4_path))

        # ===== 子图5: 震级范围误差箱线图 =====
        ax = axes[1, 1]
        magnitude_bins_box = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        bin_labels_box = [f"{magnitude_bins_box[i]:.1f}-{magnitude_bins_box[i + 1]:.1f}"
                          for i in range(len(magnitude_bins_box) - 1)]

        error_groups = []
        for i in range(len(magnitude_bins_box) - 1):
            mask = (targets >= magnitude_bins_box[i]) & (targets < magnitude_bins_box[i + 1])
            if mask.sum() > 0:
                error_groups.append(errors[mask])
            else:
                error_groups.append([])

        non_empty_groups = [group for group in error_groups if len(group) > 0]
        non_empty_labels_box = [bin_labels_box[i] for i, group in enumerate(error_groups) if len(group) > 0]

        if non_empty_groups:
            bp = ax.boxplot(non_empty_groups, labels=non_empty_labels_box, patch_artist=True)
            ax.set_xlabel('Magnitude Range', fontsize=10)
            ax.set_ylabel('Prediction Error', fontsize=10)
            ax.set_title('Error by Magnitude Range', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(True, alpha=0.3)

        # 单独保存子图5
        fig5 = plt.figure(figsize=(10, 6))
        ax5 = fig5.add_subplot(111)
        if non_empty_groups:
            bp5 = ax5.boxplot(non_empty_groups, labels=non_empty_labels_box, patch_artist=True)
            # 为箱线图添加颜色
            colors_box = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
            for patch, color in zip(bp5['boxes'], colors_box[:len(bp5['boxes'])]):
                patch.set_facecolor(color)

            # 添加质量等级参考线
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Excellent')
            ax5.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Good')
            ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Acceptable')

            ax5.set_xlabel('Magnitude Range', fontsize=12)
            ax5.set_ylabel('Prediction Error', fontsize=12)
            ax5.set_title('Error by Magnitude Range', fontsize=14, fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        plot5_path = individual_plots_dir / f'05_error_by_magnitude_{timestamp}.png'
        fig5.savefig(plot5_path, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        individual_plot_paths.append(str(plot5_path))

        # ===== 子图6: 残差图 =====
        ax = axes[1, 2]
        residuals = predictions - targets
        ax.scatter(predictions, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Magnitude', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.set_title('Residual Plot', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 单独保存子图6
        fig6 = plt.figure(figsize=(10, 6))
        ax6 = fig6.add_subplot(111)
        ax6.scatter(predictions, residuals, alpha=0.6, s=30)
        ax6.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Line')
        ax6.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='±0.2 Error')
        ax6.axhline(y=-0.2, color='green', linestyle=':', alpha=0.7)
        ax6.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='±0.3 Error')
        ax6.axhline(y=-0.3, color='orange', linestyle=':', alpha=0.7)
        ax6.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='±0.5 Error')
        ax6.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7)
        ax6.set_xlabel('Predicted Magnitude', fontsize=12)
        ax6.set_ylabel('Residuals', fontsize=12)
        ax6.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plot6_path = individual_plots_dir / f'06_residual_plot_{timestamp}.png'
        fig6.savefig(plot6_path, dpi=300, bbox_inches='tight')
        plt.close(fig6)
        individual_plot_paths.append(str(plot6_path))

        plt.tight_layout()

        # 保存总览图
        plot_path = self.exp_dir / 'plots' / f'evaluation_results_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 打印保存信息
        print(f"[INFO] Evaluation plots saved:")
        print(f"  - Overview plot: {plot_path}")
        print(f"  - Individual plots ({len(individual_plot_paths)} files): {individual_plots_dir}")
        for i, path in enumerate(individual_plot_paths, 1):
            print(f"    {i}. {Path(path).name}")

        return str(plot_path)

    def _save_prediction_plots(self, predictions, targets, timestamp):
        errors = np.abs(predictions - targets)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        ax.scatter(targets, errors, alpha=0.6, s=20)
        ax.set_xlabel('True Magnitude')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error vs Magnitude')

        z = np.polyfit(targets, errors, 1)
        p = np.poly1d(z)
        ax.plot(targets, p(targets), "r--", alpha=0.8,
                label=f'Trend: slope={z[0]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Good')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Acceptable')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        sample_indices = range(len(errors))
        ax.plot(sample_indices, errors, alpha=0.7, linewidth=1)
        ax.axhline(y=errors.mean(), color='red', linestyle='--',
                   label=f'Mean: {errors.mean():.3f}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Trend by Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')

        mae = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)

        excellent_rate = (errors <= 0.2).mean()
        good_rate = (errors <= 0.3).mean()
        acceptable_rate = (errors <= 0.5).mean()

        stats_text = f"""
        Performance Statistics
        ========================

        Regression Metrics:
        MAE:  {mae:.4f}
        RMSE: {rmse:.4f}
        R²:   {r2:.4f}

        Quality Rates:
        Excellent (≤0.2): {excellent_rate:.1%}
        Good (≤0.3):      {good_rate:.1%}
        Acceptable (≤0.5): {acceptable_rate:.1%}

        Sample Count: {len(predictions)}
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        plot_path = self.exp_dir / 'plots' / f'prediction_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def _save_association_plots(self, association_results, timestamp):

        return None

    def generate_experiment_report(self):
        """生成实验报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.exp_dir / 'reports' / f'experiment_report_{timestamp}.md'
        eval_results = self.experiment_log.get('evaluation_results', {})

        report_content = f"""# 震级预测实验报告

## 实验信息
- **实验名称**: {self.experiment_name}
- **开始时间**: {self.experiment_log['start_time']}
- **完成时间**: {datetime.datetime.now().isoformat()}
- **状态**: {self.experiment_log['status']}

## 模型信息
"""

        if 'model_info' in self.experiment_log:
            model_info = self.experiment_log['model_info']
            report_content += f"""
- **模型参数数量**: {model_info.get('total_parameters', 'N/A'):,}
- **模型大小**: {model_info.get('model_size_mb', 'N/A'):.1f} MB
- **模型特性**: {model_info.get('features', [])}
"""

        # 添加训练信息
        if 'training_history' in self.experiment_log:
            training = self.experiment_log['training_history']
            report_content += f"""
## 训练信息
- **训练轮数**: {training.get('total_epochs', 'N/A')}
- **最佳验证损失**: {training.get('best_val_loss', 'N/A'):.4f}
"""

        # 添加评估结果
        for eval_name, results in eval_results.items():
            if 'regression_metrics' in results:
                metrics = results['regression_metrics']
                report_content += f"""
## {eval_name.replace('_', ' ').title()} 结果

### 回归指标
- **MAE**: {metrics.get('mae', 'N/A'):.4f}
- **RMSE**: {metrics.get('rmse', 'N/A'):.4f}
- **R²**: {metrics.get('r2', 'N/A'):.4f}
"""

            if 'association_results' in results:
                assoc = results['association_results']
                if 'quality_metrics' in assoc:
                    quality = assoc['quality_metrics']
                    report_content += f"""
### 质量等级分析
- **优秀预测率 (≤0.2)**: {quality.get('excellent', {}).get('recall', 'N/A'):.1%}
- **良好预测率 (≤0.3)**: {quality.get('good', {}).get('recall', 'N/A'):.1%}
- **可接受预测率 (≤0.5)**: {quality.get('acceptable', {}).get('recall', 'N/A'):.1%}
"""

        # 添加文件路径信息
        report_content += f"""
## 实验文件
"""

        for file_type, file_path in self.experiment_log.get('files', {}).items():
            report_content += f"- **{file_type}**: `{file_path}`\n"

        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"实验报告已生成: {report_path}")
        return report_path

    def finalize_experiment(self, status="completed"):
        """结束实验并保存完整日志"""
        self.experiment_log['status'] = status
        self.experiment_log['end_time'] = datetime.datetime.now().isoformat()

        log_path = self.exp_dir / 'experiment_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)

        report_path = self.generate_experiment_report()

        return {
            'experiment_dir': str(self.exp_dir),
            'report_path': str(report_path),
            'log_path': str(log_path)
        }

    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

def setup_experiment_logger(experiment_name=None):
    """设置实验记录器"""
    return ExperimentLogger(experiment_name=experiment_name)


def save_complete_experiment(logger, model, trainer, results, optimizer=None):
    """保存完整实验结果的便捷函数"""
    config = {
        'model_architecture': str(model),
        'training_parameters': {
            'learning_rate': getattr(trainer, 'learning_rate', 'N/A'),
            'batch_size': getattr(trainer, 'batch_size', 'N/A'),
            'epochs': len(getattr(trainer, 'train_losses', [])),
        }
    }
    logger.save_config(config)
    logger.save_training_history(trainer)
    metrics = results.get('regression_metrics', {}) if isinstance(results, dict) else {}
    logger.save_model(model, optimizer,
                      epoch=len(getattr(trainer, 'train_losses', [])),
                      metrics=metrics)
    logger.save_evaluation_results(results)
    predictions = results.get('predictions') if isinstance(results, dict) else None
    targets = results.get('targets') if isinstance(results, dict) else None
    logger.save_plots(trainer=trainer, results=results,
                      predictions=predictions, targets=targets)
    final_info = logger.finalize_experiment()

    print("Complete experiment saved!")
    return final_info

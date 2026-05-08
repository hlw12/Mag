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

plt.rcParams.update({
    'font.size':        20,
    'axes.titlesize':   22,
    'axes.titleweight': 'bold',
    'axes.labelsize':   20,
    'axes.labelweight': 'bold',
    'xtick.labelsize':  18,
    'ytick.labelsize':  18,
    'legend.fontsize':  18,
})


class ExperimentLogger:
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
        directories = ['models', 'checkpoints', 'logs', 'plots', 'data', 'configs', 'results', 'reports']
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
            ax.text(0.5, 0.5, 'No LR History Available', ha='center', va='center', transform=ax.transAxes)
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
            ax.text(0.5, 0.5, 'No Gradient Norm History', ha='center', va='center', transform=ax.transAxes)
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
        residuals = predictions - targets  # signed

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        individual_plots_dir = self.exp_dir / 'plots' / 'individual'
        individual_plots_dir.mkdir(exist_ok=True)
        individual_plot_paths = []

        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        x_range = np.linspace(min_val, max_val, 100)

        # ===== 子图1 overview: 占位提示 =====
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.5,
                'Error Distribution by Magnitude Range\n(see individual plots 01a–01d)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, style='italic', color='gray')
        ax.set_title('Error Distribution by Range')

        range_configs = [
            (0.5, 1.5, '01a'),
            (1.5, 2.5, '01b'),
            (2.5, 3.5, '01c'),
            (3.5, 4.5, '01d'),
        ]
        for lo, hi, tag in range_configs:
            mask  = (targets >= lo) & (targets < hi)
            e_sub = residuals[mask]
            n     = mask.sum()

            fig_r, ax_r = plt.subplots(figsize=(8, 6))
            ax_r.hist(e_sub, bins=30, color='#3182bd', alpha=0.80,
                      edgecolor='#08519c', linewidth=1.0)
            ax_r.axvline(x=0,            color='#e34a33', linestyle='-',  linewidth=2.0,
                         label='Zero error')
            ax_r.axvline(x=e_sub.mean(), color='#f4a400', linestyle='--', linewidth=2.0,
                         label=f'Mean = {e_sub.mean():.3f}')
            ax_r.axvline(x= 0.2, color='green',  linestyle=':', linewidth=1.5, alpha=0.8, label='±0.2')
            ax_r.axvline(x=-0.2, color='green',  linestyle=':', linewidth=1.5, alpha=0.8)
            ax_r.axvline(x= 0.3, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='±0.3')
            ax_r.axvline(x=-0.3, color='orange', linestyle=':', linewidth=1.5, alpha=0.8)
            ax_r.set_xlabel('Prediction Error ($\\hat{y} - y$)')
            ax_r.set_ylabel('Number of Samples')
            # ax_r.set_title(
            #     f'Error Distribution  $M_L \\in [{lo},\\ {hi})$\n'
            #     f'n = {n},  MAE = {np.abs(e_sub).mean():.3f},  '
            #     f'RMSE = {np.sqrt((e_sub**2).mean()):.3f}'
            # )
            ax_r.legend()
            ax_r.grid(True, alpha=0.35, axis='y', linestyle='--')
            plt.tight_layout()
            plot_r_path = individual_plots_dir / (
                f'{tag}_error_dist_mag_'
                f'{str(lo).replace(".","p")}_{str(hi).replace(".","p")}_{timestamp}.png'
            )
            fig_r.savefig(plot_r_path, dpi=300, bbox_inches='tight')
            plt.close(fig_r)
            individual_plot_paths.append(str(plot_r_path))

        # ===== 子图2 overview: 混淆矩阵 =====
        ax = axes[0, 1]
        if 'association_results' in results and 'confusion_matrix' in results['association_results']['classification_metrics']:
            magnitude_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            bin_labels = [f"{magnitude_bins[i]:.1f}-{magnitude_bins[i + 1]:.1f}"
                          for i in range(len(magnitude_bins) - 1)]
            cm = results['association_results']['classification_metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=bin_labels, yticklabels=bin_labels,
                        ax=ax, annot_kws={'size': 10}, cbar_kws={'shrink': 0.8})
            ax.set_title('Magnitude Range Confusion Matrix')
            ax.set_xlabel('Predicted Magnitude Range')
            ax.set_ylabel('True Magnitude Range')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

            # ===== 子图2 独立保存 =====
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=bin_labels, yticklabels=bin_labels,
                        ax=ax2, annot_kws={'size': 18}, cbar_kws={'shrink': 0.8})
            ax2.set_title('Magnitude Range Confusion Matrix')
            ax2.set_xlabel('Predicted Magnitude Range')
            ax2.set_ylabel('True Magnitude Range')
            ax2.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='y', rotation=0)
            plt.tight_layout()
            plot2_path = individual_plots_dir / f'02_confusion_matrix_{timestamp}.png'
            fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            individual_plot_paths.append(str(plot2_path))
        else:
            ax.text(0.5, 0.5, 'No Confusion Matrix Data Available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confusion Matrix (No Data)')

        # ===== 子图3 overview: 误差分布直方图 =====
        ax = axes[0, 2]
        ax.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='Excellent (0.2)')
        ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Good (0.3)')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Acceptable (0.5)')
        ax.axvline(x=errors.mean(), color='purple', linestyle='-', linewidth=2,
                   label=f'Mean ({errors.mean():.3f})')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ===== 子图3 独立保存 =====
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hist(errors, bins=50, alpha=0.7, color='skyblue',
                 edgecolor='black', linewidth=0.5, density=True)
        ax3.axvline(x=0.2, color='green', linestyle='--', linewidth=3, label='Excellent (0.2)')
        ax3.axvline(x=0.3, color='orange', linestyle='--', linewidth=3, label='Good (0.3)')
        ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=3, label='Acceptable (0.5)')
        ax3.axvline(x=errors.mean(), color='purple', linestyle='-', linewidth=2,
                    label=f'Mean ({errors.mean():.3f})')
        ax3.axvline(x=np.median(errors), color='brown', linestyle='-', linewidth=2,
                    label=f'Median ({np.median(errors):.3f})')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        stats_text = f'Statistics:\nStd Dev: {errors.std():.3f}\nMax Error: {errors.max():.3f}\nSamples: {len(errors)}'
        ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=16,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plot3_path = individual_plots_dir / f'03_error_distribution_{timestamp}.png'
        fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        individual_plot_paths.append(str(plot3_path))

        # ===== 子图4 overview: 质量分布饼图 =====
        ax = axes[1, 0]
        excellent_count  = (errors <= 0.2).sum()
        good_count       = ((errors > 0.2) & (errors <= 0.3)).sum()
        acceptable_count = ((errors > 0.3) & (errors <= 0.5)).sum()
        poor_count       = (errors > 0.5).sum()
        counts  = [excellent_count, good_count, acceptable_count, poor_count]
        labels  = ['Excellent\n(≤0.2)', 'Good\n(0.2-0.3)', 'Acceptable\n(0.3-0.5)', 'Poor\n(>0.5)']
        colors  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        non_zero_mask = np.array(counts) > 0
        if non_zero_mask.any():
            filtered_counts = np.array(counts)[non_zero_mask]
            filtered_labels = np.array(labels)[non_zero_mask]
            filtered_colors = np.array(colors)[non_zero_mask]
            ax.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        ax.set_title('Prediction Quality Distribution')

        # ===== 子图4 独立保存 =====
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        if non_zero_mask.any():
            wedges4, _, _ = ax4.pie(filtered_counts, labels=filtered_labels,
                                    colors=filtered_colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 18})
            ax4.legend(wedges4,
                       [f'{label}: {count}' for label, count in zip(filtered_labels, filtered_counts)],
                       title="Sample Statistics", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax4.set_title('Prediction Quality Distribution')
        plt.tight_layout()
        plot4_path = individual_plots_dir / f'04_quality_distribution_{timestamp}.png'
        fig4.savefig(plot4_path, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        individual_plot_paths.append(str(plot4_path))

        # ===== 子图5 overview: 震级范围误差箱线图 =====
        ax = axes[1, 1]
        magnitude_bins_box = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        bin_labels_box = [f"{magnitude_bins_box[i]:.1f}-{magnitude_bins_box[i + 1]:.1f}"
                          for i in range(len(magnitude_bins_box) - 1)]
        error_groups = []
        for i in range(len(magnitude_bins_box) - 1):
            mask = (targets >= magnitude_bins_box[i]) & (targets < magnitude_bins_box[i + 1])
            error_groups.append(errors[mask] if mask.sum() > 0 else [])
        non_empty_groups     = [g for g in error_groups if len(g) > 0]
        non_empty_labels_box = [bin_labels_box[i] for i, g in enumerate(error_groups) if len(g) > 0]
        if non_empty_groups:
            ax.boxplot(non_empty_groups, labels=non_empty_labels_box, patch_artist=True)
            ax.set_xlabel('Magnitude Range')
            ax.set_ylabel('Prediction Error')
            ax.set_title('Error by Magnitude Range')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(True, alpha=0.3)

        # ===== 子图5 独立保存 =====
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        if non_empty_groups:
            bp5 = ax5.boxplot(non_empty_groups, labels=non_empty_labels_box, patch_artist=True)
            colors_box = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
            for patch, color in zip(bp5['boxes'], colors_box[:len(bp5['boxes'])]):
                patch.set_facecolor(color)
            ax5.axhline(y=0.2, color='green',  linestyle='--', alpha=0.7, label='Excellent')
            ax5.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Good')
            ax5.axhline(y=0.5, color='red',    linestyle='--', alpha=0.7, label='Acceptable')
            ax5.set_xlabel('Magnitude Range')
            ax5.set_ylabel('Prediction Error')
            ax5.set_title('Error by Magnitude Range')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plot5_path = individual_plots_dir / f'05_error_by_magnitude_{timestamp}.png'
        fig5.savefig(plot5_path, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        individual_plot_paths.append(str(plot5_path))

        ax = axes[1, 2]
        ax.scatter(predictions, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Magnitude')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.scatter(predictions, residuals, alpha=0.6, s=30)
        ax6.axhline(y=0,    color='r',      linestyle='--', linewidth=2, label='Zero Line')
        ax6.axhline(y=0.2,  color='green',  linestyle=':',  alpha=0.7,  label='±0.2 Error')
        ax6.axhline(y=-0.2, color='green',  linestyle=':',  alpha=0.7)
        ax6.axhline(y=0.3,  color='orange', linestyle=':',  alpha=0.7,  label='±0.3 Error')
        ax6.axhline(y=-0.3, color='orange', linestyle=':',  alpha=0.7)
        ax6.axhline(y=0.5,  color='red',    linestyle=':',  alpha=0.7,  label='±0.5 Error')
        ax6.axhline(y=-0.5, color='red',    linestyle=':',  alpha=0.7)
        ax6.set_xlabel('Predicted Magnitude')
        ax6.set_ylabel('Residuals')
        ax6.set_title('Residual Analysis')
        ax6.legend(fontsize=12)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plot6_path = individual_plots_dir / f'06_residual_plot_{timestamp}.png'
        fig6.savefig(plot6_path, dpi=300, bbox_inches='tight')
        plt.close(fig6)
        individual_plot_paths.append(str(plot6_path))

        plt.tight_layout()
        plot_path = self.exp_dir / 'plots' / f'evaluation_results_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

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
        ax.plot(targets, p(targets), "r--", alpha=0.8, label=f'Trend: slope={z[0]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axvline(x=0.2, color='green',  linestyle='--', alpha=0.7, label='Excellent')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Good')
        ax.axvline(x=0.5, color='red',    linestyle='--', alpha=0.7, label='Acceptable')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.plot(range(len(errors)), errors, alpha=0.7, linewidth=1)
        ax.axhline(y=errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.3f}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Trend by Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.axis('off')
        mae  = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        r2   = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
        excellent_rate  = (errors <= 0.2).mean()
        good_rate       = (errors <= 0.3).mean()
        acceptable_rate = (errors <= 0.5).mean()
        stats_text = (
            f"  Performance Statistics\n"
            f"  ========================\n\n"
            f"  Regression Metrics:\n"
            f"  MAE:  {mae:.4f}\n"
            f"  RMSE: {rmse:.4f}\n"
            f"  R\u00b2:   {r2:.4f}\n\n"
            f"  Quality Rates:\n"
            f"  Excellent (\u22640.2): {excellent_rate:.1%}\n"
            f"  Good (\u22640.3):      {good_rate:.1%}\n"
            f"  Acceptable (\u22640.5): {acceptable_rate:.1%}\n\n"
            f"  Sample Count: {len(predictions)}"
        )
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=16,
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
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            report_content += (
                f"\n- **模型参数数量**: {model_info.get('total_parameters', 'N/A')}\n"
                f"- **模型大小**: {model_info.get('model_size_mb', 'N/A'):.1f} MB\n"
                f"- **模型特性**: {model_info.get('features', [])}\n"
            )
        if 'training_history' in self.experiment_log:
            training = self.experiment_log['training_history']
            report_content += (
                f"\n## 训练信息\n"
                f"- **训练轮数**: {training.get('total_epochs', 'N/A')}\n"
                f"- **最佳验证损失**: {training.get('best_val_loss', 'N/A'):.4f}\n"
            )
        for eval_name, res in eval_results.items():
            if 'regression_metrics' in res:
                metrics = res['regression_metrics']
                report_content += (
                    f"\n## {eval_name.replace('_', ' ').title()} 结果\n\n"
                    f"### 回归指标\n"
                    f"- **MAE**: {metrics.get('mae', 'N/A'):.4f}\n"
                    f"- **RMSE**: {metrics.get('rmse', 'N/A'):.4f}\n"
                    f"- **R²**: {metrics.get('r2', 'N/A'):.4f}\n"
                )
            if 'association_results' in res:
                assoc = res['association_results']
                if 'quality_metrics' in assoc:
                    quality = assoc['quality_metrics']
                    report_content += (
                        f"\n### 质量等级分析\n"
                        f"- **优秀预测率 (≤0.2)**: {quality.get('excellent', {}).get('recall', 'N/A'):.1%}\n"
                        f"- **良好预测率 (≤0.3)**: {quality.get('good', {}).get('recall', 'N/A'):.1%}\n"
                        f"- **可接受预测率 (≤0.5)**: {quality.get('acceptable', {}).get('recall', 'N/A'):.1%}\n"
                    )
        report_content += "\n## 实验文件\n"
        for file_type, file_path in self.experiment_log.get('files', {}).items():
            report_content += f"- **{file_type}**: `{file_path}`\n"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"实验报告已生成: {report_path}")
        return report_path

    def finalize_experiment(self, status="completed"):
        self.experiment_log['status']   = status
        self.experiment_log['end_time'] = datetime.datetime.now().isoformat()
        log_path = self.exp_dir / 'experiment_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)
        report_path = self.generate_experiment_report()
        return {
            'experiment_dir': str(self.exp_dir),
            'report_path':    str(report_path),
            'log_path':       str(log_path)
        }

    def _make_serializable(self, obj):
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
    return ExperimentLogger(experiment_name=experiment_name)


def save_complete_experiment(logger, model, trainer, results, optimizer=None):
    config = {
        'model_architecture': str(model),
        'training_parameters': {
            'learning_rate': getattr(trainer, 'learning_rate', 'N/A'),
            'batch_size':    getattr(trainer, 'batch_size', 'N/A'),
            'epochs':        len(getattr(trainer, 'train_losses', [])),
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
    targets     = results.get('targets')     if isinstance(results, dict) else None
    logger.save_plots(trainer=trainer, results=results,
                      predictions=predictions, targets=targets)
    final_info = logger.finalize_experiment()
    print("Complete experiment saved!")
    return final_info
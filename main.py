#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/29
# @Author  : 上头欢乐送、
# @File    : main.py
# @Software: PyCharm
# 学习新思想，争做新青年

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from Netgroup.Net import create_magnitude_model
from Trainer import MagnitudeTrainer, evaluate_model
from cfg.ConfigLoader import ConfigLoader, \
    print_config_summary, MagnitudeModelConfig, create_argparser
from dataset import MyDataset
from saver import ExperimentLogger, save_complete_experiment
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Fixed random seed: {seed}")


def setup_device(device_config="auto"):
    """设置计算设备"""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    print(f"[INFO] Using device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"[INFO] GPU device: {torch.cuda.get_device_name()}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    return device


def visualize_event_split(dataset, train_indices, val_indices, test_indices,
                          save_path='event_split_verification.png'):
    """
    可视化验证event-based split的无泄漏性
    """
    all_samples = dataset.getAllsample()

    # 提取各集合的event和magnitude
    train_events = [all_samples[i]['source_id'] for i in train_indices]
    val_events = [all_samples[i]['source_id'] for i in val_indices]
    test_events = [all_samples[i]['source_id'] for i in test_indices]

    train_mags = [all_samples[i]['magnitude'] for i in train_indices]
    val_mags = [all_samples[i]['magnitude'] for i in val_indices]
    test_mags = [all_samples[i]['magnitude'] for i in test_indices]

    # 转换为集合检查重叠
    train_event_set = set(train_events)
    val_event_set = set(val_events)
    test_event_set = set(test_events)

    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ============ (a) 重叠检查统计表 ============
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')

    overlap_data = [
        ['Train ∩ Val', len(train_event_set & val_event_set),
         '✓ PASS' if len(train_event_set & val_event_set) == 0 else '✗ FAIL'],
        ['Train ∩ Test', len(train_event_set & test_event_set),
         '✓ PASS' if len(train_event_set & test_event_set) == 0 else '✗ FAIL'],
        ['Val ∩ Test', len(val_event_set & test_event_set),
         '✓ PASS' if len(val_event_set & test_event_set) == 0 else '✗ FAIL'],
        ['Train ∪ Val ∪ Test', len(train_event_set | val_event_set | test_event_set),
         f'{len(all_samples)} total unique events']
    ]

    table = ax1.table(cellText=overlap_data,
                      colLabels=['Set Operation', 'Overlap Count', 'Status'],
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 标题加粗
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 结果行着色
    for i in range(1, 4):
        status = overlap_data[i - 1][2]
        if '✓' in status:
            table[(i, 2)].set_facecolor('#C8E6C9')
        else:
            table[(i, 2)].set_facecolor('#FFCDD2')

    ax1.set_title('(a) Event Overlap Verification', fontsize=14, fontweight='bold', pad=20)

    # ============ (b) 数据集统计 ============
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    stats_data = [
        ['Train', len(train_event_set), len(train_indices), f'{len(train_indices) / len(train_event_set):.2f}'],
        ['Val', len(val_event_set), len(val_indices), f'{len(val_indices) / len(val_event_set):.2f}'],
        ['Test', len(test_event_set), len(test_indices), f'{len(test_indices) / len(test_event_set):.2f}'],
    ]

    table2 = ax2.table(cellText=stats_data,
                       colLabels=['Split', 'Events', 'Traces', 'Avg'],
                       cellLoc='center', loc='center',
                       colWidths=[0.2, 0.2, 0.2, 0.2])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2.5)

    for i in range(4):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('(b) Dataset Statistics', fontsize=14, fontweight='bold', pad=20)

    # ============ (c) 事件数和trace数对比 ============
    ax3 = fig.add_subplot(gs[1, 0])

    splits = ['Train', 'Val', 'Test']
    events = [len(train_event_set), len(val_event_set), len(test_event_set)]
    traces = [len(train_indices), len(val_indices), len(test_indices)]

    x = np.arange(len(splits))
    width = 0.35

    bars1 = ax3.bar(x - width / 2, events, width, label='Events', alpha=0.8, color='#FF9800')
    bars2 = ax3.bar(x + width / 2, traces, width, label='Traces', alpha=0.8, color='#03A9F4')

    ax3.set_xlabel('Dataset Split', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('(c) Events vs Traces Distribution', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # ============ (d) 震级分布对比 ============
    ax4 = fig.add_subplot(gs[1, 1:])

    bins = np.arange(0.5, 5.0, 0.5)
    ax4.hist([train_mags, val_mags, test_mags], bins=bins,
             label=['Train', 'Val', 'Test'], alpha=0.6, edgecolor='black')
    ax4.set_xlabel('Magnitude', fontweight='bold')
    ax4.set_ylabel('Trace Count', fontweight='bold')
    ax4.set_title('(d) Magnitude Distribution Across Splits', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # ============ (e) 按震级区间统计事件分布 ============
    ax5 = fig.add_subplot(gs[2, :])

    mag_bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5),
                (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5)]

    def count_events_by_mag(events, samples, mag_range):
        event_mags = {}
        for i, sample in enumerate(samples):
            if sample['source_id'] in events:
                event_mags[sample['source_id']] = sample['magnitude']
        return sum(1 for mag in event_mags.values()
                   if mag_range[0] <= mag < mag_range[1])

    train_counts = [count_events_by_mag(train_event_set, all_samples, mb) for mb in mag_bins]
    val_counts = [count_events_by_mag(val_event_set, all_samples, mb) for mb in mag_bins]
    test_counts = [count_events_by_mag(test_event_set, all_samples, mb) for mb in mag_bins]

    x = np.arange(len(mag_bins))
    width = 0.25

    ax5.bar(x - width, train_counts, width, label='Train', alpha=0.8, color='#4CAF50')
    ax5.bar(x, val_counts, width, label='Val', alpha=0.8, color='#FFC107')
    ax5.bar(x + width, test_counts, width, label='Test', alpha=0.8, color='#F44336')

    ax5.set_xlabel('Magnitude Range', fontweight='bold')
    ax5.set_ylabel('Event Count', fontweight='bold')
    ax5.set_title('(e) Event Distribution by Magnitude Range (Stratified Split)', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'[{mb[0]},{mb[1]})' for mb in mag_bins], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 总标题
    total_events = len(train_event_set) + len(val_event_set) + len(test_event_set)
    overlap_status = "✓ ZERO OVERLAP" if (len(train_event_set & val_event_set) == 0 and
                                          len(train_event_set & test_event_set) == 0 and
                                          len(val_event_set & test_event_set) == 0) else "✗ OVERLAP DETECTED"

    fig.suptitle(
        f'Event-Based Data Split Verification | Total: {total_events} unique events | Status: {overlap_status}',
        fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Visualization saved to {save_path}")
    plt.close()

    # 打印验证报告
    print("\n" + "=" * 70)
    print("EVENT-BASED SPLIT VERIFICATION REPORT")
    print("=" * 70)
    print(f"Train: {len(train_event_set)} events, {len(train_indices)} traces")
    print(f"Val:   {len(val_event_set)} events, {len(val_indices)} traces")
    print(f"Test:  {len(test_event_set)} events, {len(test_indices)} traces")
    print("-" * 70)
    print(f"Train ∩ Val:  {len(train_event_set & val_event_set)} events")
    print(f"Train ∩ Test: {len(train_event_set & test_event_set)} events")
    print(f"Val ∩ Test:   {len(val_event_set & test_event_set)} events")
    print("-" * 70)
    if overlap_status == "✓ ZERO OVERLAP":
        print("✓ VERIFICATION PASSED: Zero event overlap confirmed!")
    else:
        print("✗ VERIFICATION FAILED: Event overlap detected!")
    print("=" * 70 + "\n")


def create_data_loaders(dataset, config_manager: MagnitudeModelConfig):
    """创建数据集"""
    dl_config = config_manager.get_dataloader_config()
    split_ratios = dl_config['split_ratios']
    all_samples = dataset.getAllsample()
    event_to_indices = {}
    for idx, sample in enumerate(all_samples):
        event_id = sample['source_id']
        if event_id not in event_to_indices:
            event_to_indices[event_id] = []
        event_to_indices[event_id].append(idx)

    unique_events = list(event_to_indices.keys())
    np.random.shuffle(unique_events)
    n_train = int(len(unique_events) * split_ratios['train'])
    n_val = int(len(unique_events) * split_ratios['val'])
    train_events = unique_events[:n_train]
    val_events = unique_events[n_train:n_train + n_val]
    test_events = unique_events[n_train + n_val:]

    train_indices = [idx for e in train_events for idx in event_to_indices[e]]
    val_indices = [idx for e in val_events for idx in event_to_indices[e]]
    test_indices = [idx for e in test_events for idx in event_to_indices[e]]

    visualize_event_split(dataset, train_indices, val_indices, test_indices,
                          save_path='experiments/event_split_verification.png')

    print(f"[INFO] Event-based split: train={len(train_events)} events/{len(train_indices)} traces, "
          f"val={len(val_events)} events/{len(val_indices)} traces, "
          f"test={len(test_events)} events/{len(test_indices)} traces")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # print(f"[INFO] Loading dataset...")
    # print(f"[INFO] CSV path: {data_config['csv_path']}")
    # print(f"[INFO] Waveform path: {data_config['wave_path']}")

    # dataset = MyDataset(
    #     csvPath=data_config['csv_path'],
    #     wavePath=data_config['wave_path'],
    #     window_size = data_config['window_samples'],
    #     mode='train',
    #     filter_params=data_config['filter_params'],
    #     sampling_by_magnitude=data_config['sampling_by_magnitude']
    # )
    common_kwargs = {
        'batch_size': dl_config['batch_size'],
        'num_workers': dl_config['num_workers'],
        'pin_memory': dl_config['pin_memory'],
        'drop_last': dl_config.get('drop_last', False)
    }

    train_loader = DataLoader(train_dataset, shuffle=dl_config['shuffle']['train'], **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=dl_config['shuffle']['val'], **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=dl_config['shuffle']['test'], **common_kwargs)
    print(f"[INFO] Dataset loaded successfully, total {len(dataset)} samples")

    return train_loader, val_loader, test_loader


def create_dataset(config_manager: MagnitudeModelConfig):
    """创建完整数据集"""
    data_config = config_manager.get_data_config()

    print(f"[INFO] Loading dataset...")
    dataset = MyDataset(
        csvPath=data_config['csv_path'],
        wavePath=data_config['wave_path'],
        window_size=data_config['window_samples'],
        filter_params=data_config['filter_params'],
        sampling_by_magnitude=data_config['sampling_by_magnitude'],
    )
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    return dataset


def create_model_from_config(config_manager: MagnitudeModelConfig):
    """从配置创建模型"""
    model_config = config_manager.get_model_config()

    print(f"[INFO] Creating model...")
    print(f"[INFO] Model config: input_channels={model_config['input_channels']}, "
          f"hidden_dim={model_config['hidden_dim']}, dropout={model_config['dropout']}")

    model = create_magnitude_model(
        input_channels=model_config['input_channels'],
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    )

    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"[INFO] Model parameters: {model_info['total_parameters']:,}")
        print(f"[INFO] Model size: {model_info['model_size_mb']:.1f} MB")

    return model


def setup_experiment_logger(config_manager: MagnitudeModelConfig):
    """设置实验记录器"""
    logging_config = config_manager.get_logging_config()
    experiment_config = config_manager.get_experiment_config()

    if not logging_config['enable']:
        print("[INFO] Experiment logging disabled")
        return None

    logger = ExperimentLogger(
        base_dir=logging_config['experiment_logger'].get('base_dir', 'experiments'),
        experiment_name=experiment_config['name']
    )

    config_save_path = logger.exp_dir / 'configs' / 'full_config.yaml'
    loader = ConfigLoader(config_path='cfg/base.yaml')
    loader.save_config(config_manager.config, config_save_path)

    return logger


def run_magnitude_training(config_path="cfg/base.yaml", args=None, updates=None, **kwargs):
    """
    运行震级预测模型训练

    Args:
        config_path: 配置文件路径
        args: 命令行参数
        updates: 配置更新字典
        **kwargs: 其他参数

    Returns:
        dict: 包含训练结果的字典
    """
    print("[INFO] Starting magnitude prediction model training...")
    config_manager = MagnitudeModelConfig(config_path)
    config_manager.load_config(args=args, updates=updates)
    print_config_summary(config_manager.config)
    experiment_config = config_manager.get_experiment_config()
    set_seed(experiment_config['random_seed'])
    device = setup_device(experiment_config['device'])
    logger = setup_experiment_logger(config_manager)

    try:
        dataset = create_dataset(config_manager)
        train_loader, val_loader, test_loader = create_data_loaders(dataset, config_manager)
        model = create_model_from_config(config_manager)
        model = model.to(device)
        trainer = MagnitudeTrainer(model, device=device)
        training_config = config_manager.get_training_config()

        print(f"[INFO] Starting training...")
        print(f"[INFO] Training params: epochs={training_config['num_epochs']}, "
              f"lr={training_config['learning_rate']}, "
              f"patience={training_config['early_stopping']['patience']}")

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=training_config['num_epochs'],
            lr=training_config['learning_rate'],
            patience=training_config['early_stopping']['patience'],
            save_path=training_config['model_saving']['save_path']
        )

        eval_config = config_manager.get_evaluation_config()
        if eval_config.get('visualization', {}).get('enable', True):
            trainer.plot_training_history()
        print("[INFO] Loading best model for evaluation...")
        checkpoint = torch.load(training_config['model_saving']['save_path'],
                                map_location=device,
                                weights_only=False
                                )
        model.load_state_dict(checkpoint['model_state_dict'])
        results = evaluate_model(model, test_loader, device)
        if logger is not None:
            save_complete_experiment(logger, model, trainer, results)
            print("[INFO] Experiment results saved completely!")

        return results

    except Exception as e:
        print(f"[ERROR] Error occurred during training: {e}")
        if logger is not None:
            logger.finalize_experiment(status="failed")
        raise

    finally:
        if 'dataset' in locals():
            dataset.close()


def run_magnitude_testing(config_path="cfg/base.yaml", model_path='best_model.pt',
                          args=None, updates=None):
    print("[INFO] Starting magnitude prediction model testing...")
    config_manager = MagnitudeModelConfig(config_path)
    config_manager.load_config(args=args, updates=updates)
    print_config_summary(config_manager.config)
    experiment_config = config_manager.get_experiment_config()
    set_seed(experiment_config['random_seed'])
    device = setup_device(experiment_config['device'])
    logger = setup_experiment_logger(config_manager)

    try:
        dataset = create_dataset(config_manager)
        _, _, test_loader = create_data_loaders(dataset, config_manager)
        model = create_model_from_config(config_manager)
        model = model.to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False )
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = MagnitudeTrainer(model, device=device)
        results = evaluate_model(model, test_loader, device)
        print(results.keys())
        if logger is not None:
            save_complete_experiment(logger, model, trainer, results)
        return results

    except Exception as e:
        print(f"[ERROR] Error occurred during testing: {e}")
        raise

def load_cfg(config_path="cfg/base.yaml", args=None):
    """
    加载配置文件

    Args:
        config_path: 配置文件路径
        args: 命令行参数或更新字典

    Returns:
        dict: 加载的配置字典
    """
    try:
        config_manager = MagnitudeModelConfig(config_path)
        config = config_manager.load_config(args=args)
        print("[INFO] Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        raise


def create_training_from_yaml(config_path="cfg/base.yaml",
                              experiment_name=None,
                              **overrides):
    """
    从YAML配置文件创建训练任务的便捷函数

    Args:
        config_path: 配置文件路径
        experiment_name: 实验名称（覆盖配置文件中的设置）
        **overrides: 其他要覆盖的配置参数

    Returns:
        训练结果
    """
    updates = {}
    if experiment_name:
        updates['experiment'] = {'name': experiment_name}

    for key, value in overrides.items():
        keys = key.split('.')
        current = updates
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return run_magnitude_training(
        config_path=config_path,
        updates=updates
    )

# 训练/测试/验证
if __name__ == "__main__":
    import sys
    parser = create_argparser()
    args = parser.parse_args()

    try:
        results = run_magnitude_training(
            config_path=args.config,
            args=args
        )

        if 'regression_metrics' in results:
            metrics = results['regression_metrics']
            print(f"\n[Result Summary]")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")

    except Exception as e:
        print(f"[ERROR] Program execution failed: {e}")
        sys.exit(1)

# if __name__ == "__main__":
#     run_magnitude_testing(config_path="cfg/base.yaml",
#                           model_path='experiments/magnitude_evaluate_experiment_500_norm_std/checkpoints/best_model.pth')
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


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 固定随机种子: {seed}")


def setup_device(device_config="auto"):
    """设置计算设备"""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    print(f"[INFO] 使用设备: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"[INFO] GPU设备: {torch.cuda.get_device_name()}")
        print(f"[INFO] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    return device


def create_dataset(config_manager: MagnitudeModelConfig):
    """创建数据集"""
    data_config = config_manager.get_data_config()

    print(f"[INFO] 加载数据集...")
    print(f"[INFO] CSV路径: {data_config['csv_path']}")
    print(f"[INFO] 波形路径: {data_config['wave_path']}")

    dataset = MyDataset(
        csvPath=data_config['csv_path'],
        wavePath=data_config['wave_path'],
        window_size = data_config['window_samples'],
        mode='train',
        filter_params=data_config['filter_params'],
        sampling_by_magnitude=data_config['sampling_by_magnitude'],
    )

    print(f"[INFO] 数据集加载完成，共{len(dataset)}个样本")
    return dataset


def create_data_loaders(dataset, config_manager: MagnitudeModelConfig):
    """创建数据加载器"""
    dl_config = config_manager.get_dataloader_config()

    total_size = len(dataset)
    split_ratios = dl_config['split_ratios']

    perm = torch.randperm(total_size)
    train_size = int(split_ratios['train'] * total_size)
    val_size = int(split_ratios['val'] * total_size)

    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"[INFO] 数据分割: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}, 测试集{len(test_dataset)}")
    common_kwargs = {
        'batch_size': dl_config['batch_size'],
        'num_workers': dl_config['num_workers'],
        'pin_memory': dl_config['pin_memory'],
        'drop_last': dl_config.get('drop_last', False)
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=dl_config['shuffle']['train'],
        **common_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=dl_config['shuffle']['val'],
        **common_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=dl_config['shuffle']['test'],
        **common_kwargs
    )

    return train_loader, val_loader, test_loader


def create_model_from_config(config_manager: MagnitudeModelConfig):
    """从配置创建模型"""
    model_config = config_manager.get_model_config()

    print(f"[INFO] 创建模型...")
    print(f"[INFO] 模型配置: 输入通道={model_config['input_channels']}, "
          f"隐藏维度={model_config['hidden_dim']}, Dropout={model_config['dropout']}")

    model = create_magnitude_model(
        input_channels=model_config['input_channels'],
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    )

    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"[INFO] 模型参数数量: {model_info['total_parameters']:,}")
        print(f"[INFO] 模型大小: {model_info['model_size_mb']:.1f} MB")

    return model


def setup_experiment_logger(config_manager: MagnitudeModelConfig):
    """设置实验记录器"""
    logging_config = config_manager.get_logging_config()
    experiment_config = config_manager.get_experiment_config()

    if not logging_config['enable']:
        print("[INFO] 实验日志记录已禁用")
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
    print("[INFO] 开始震级预测模型训练...")
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

        print(f"[INFO] 开始训练...")
        print(f"[INFO] 训练参数: 轮数={training_config['num_epochs']}, "
              f"学习率={training_config['learning_rate']}, "
              f"耐心值={training_config['early_stopping']['patience']}")

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
        print("[INFO] 加载最佳模型进行评估...")
        checkpoint = torch.load(training_config['model_saving']['save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        results = evaluate_model(model, test_loader, device)
        if logger is not None:
            save_complete_experiment(logger, model, trainer, results)
            print("[INFO] 实验结果已完整保存!")

        return results

    except Exception as e:
        print(f"[ERROR] 训练过程中发生错误: {e}")
        if logger is not None:
            logger.finalize_experiment(status="failed")
        raise

    finally:
        if 'dataset' in locals():
            dataset.close()


def run_magnitude_testing(config_path="cfg/base.yaml", model_path='best_model.pt',
                          args=None, updates=None):
    print("[INFO] 开始震级预测模型测试...")
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
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = MagnitudeTrainer(model, device=device)
        results = evaluate_model(model, test_loader, device)
        print(results.keys())
        if logger is not None:
            save_complete_experiment(logger, model, trainer, results)
        return results

    except Exception as e:
        print(f"[ERROR] 测试过程中发生错误: {e}")
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
        print("[INFO] 配置加载成功")
        return config
    except Exception as e:
        print(f"[ERROR] 配置加载失败: {e}")
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
            print(f"\n[结果摘要]")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")

    except Exception as e:
        print(f"[ERROR] Program execution failed: {e}")
        sys.exit(1)
# if __name__ == "__main__":
#     run_magnitude_testing(config_path="cfg/base.yaml",
#                           model_path='experiments/magnitude_prediction_experiment_500_chunk3/checkpoints/best_model.pth')
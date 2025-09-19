#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/4 00:07
# @Author  : 上头欢乐送、
# @File    : ConfigLoader.py
# @Software: PyCharm
# 学习新思想，争做新青年

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import argparse
from dataclasses import dataclass, field
from copy import deepcopy

class MagnitudeModelConfig:
    """Magnitude Prediction Model Configuration Manager"""

    def __init__(self, config_path: str = "cfg/base.yaml"):
        self.config_path = config_path
        self.config = None

    def load_config(self, args=None, updates=None):
        """Load configuration file"""
        try:
            self.config = load_and_merge_config(
                config_path=self.config_path,
                args=args,
                updates=updates
            )
            validate_config(self.config)
            return self.config
        except Exception as e:
            print(f"[ERROR] Configuration loading failed: {e}")
            raise

    def get_data_config(self):
        """Get data configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded, please call load_config() first")

        data_config = self.config['data']

        # Build sampling configuration
        sampling_config = {}
        if data_config.get('sampling_by_magnitude', {}).get('enable', True):
            for sample in data_config['sampling_by_magnitude']['samples']:
                range_tuple = tuple(sample['range'])
                sampling_config[range_tuple] = sample['count']

        return {
            'csv_path': data_config['paths']['csv_path'],
            'wave_path': data_config['paths']['wave_path'],
            'filter_params': data_config['filter_params'],
            'sampling_by_magnitude': sampling_config if sampling_config else None,
            'window_samples': data_config['preprocessing']['window_samples'],
            'target_length': data_config['preprocessing']['target_length'],
            'spec_config': {
                'n_fft': data_config['spectrogram']['n_fft'],
                'win_length': data_config['spectrogram']['win_length'],
                'hop_length': data_config['spectrogram']['hop_length'],
                'power': data_config['spectrogram']['power']
            }
        }

    def get_dataloader_config(self):
        """Get data loader configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")

        dl_config = self.config['dataloader']
        return {
            'split_ratios': dl_config['split_ratios'],
            'batch_size': dl_config['batch_size'],
            'num_workers': dl_config['num_workers'],
            'pin_memory': dl_config['pin_memory'],
            'shuffle': dl_config['shuffle'],
            'drop_last': dl_config.get('drop_last', False)
        }

    def get_model_config(self):
        """Get model configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")

        return self.config['model']

    def get_training_config(self):
        """Get training configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")
        training_config = self.config['training'].copy()
        experiment_name = self.config['experiment']['name']
        original_save_path = training_config['model_saving']['save_path']
        training_config['model_saving']['save_path'] = f"experiments/{experiment_name}/{original_save_path}"
        return training_config

    def get_experiment_config(self):
        """Get experimental configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")

        return self.config['experiment']

    def get_logging_config(self):
        """Get log configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")
        return self.config['logging']

    def get_evaluation_config(self):
        """Get the evaluation configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded!")
        return self.config.get('evaluation', {})



class ConfigLoader:
    """Configuration file loader"""
    def __init__(self, config_path: str = "base.yaml"):
        self.config_path = Path(config_path)
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"[INFO] Configuration file loaded successfully: {self.config_path}")
            return self.config
        except yaml.YAMLError as e:
            raise ValueError(f"YAML configuration file parsing error: {e}")
        except Exception as e:
            raise RuntimeError(f"Configuration file loading failed: {e}")

    def save_config(self, config: Dict[str, Any], save_path: str = None):
        """Save the configuration to a YAML file"""
        if save_path is None:
            save_path = self.config_path
        else:
            save_path = Path(save_path)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False,
                          allow_unicode=True, indent=2)
            print(f"[INFO] Configuration file saved successfully: {save_path}")
        except Exception as e:
            raise RuntimeError(f"Configuration file save failed: {e}")

    def update_config_from_args(self, config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """从命令行参数更新配置"""
        updated_config = deepcopy(config)

        # # 实验配置更新
        # if hasattr(args, 'experiment_name') and args.experiment_name:
        #     updated_config['experiment']['name'] = args.experiment_name
        if hasattr(args, 'seed') and args.seed is not None:
            updated_config['experiment']['random_seed'] = args.seed
        if hasattr(args, 'device') and args.device:
            updated_config['experiment']['device'] = args.device

        # 数据配置更新
        if hasattr(args, 'csv_path') and args.csv_path:
            updated_config['data']['paths']['csv_path'] = args.csv_path
        if hasattr(args, 'wave_path') and args.wave_path:
            updated_config['data']['paths']['wave_path'] = args.wave_path
        if hasattr(args, 'batch_size') and args.batch_size:
            updated_config['dataloader']['batch_size'] = args.batch_size
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            updated_config['dataloader']['num_workers'] = args.num_workers

        # 模型配置更新
        if hasattr(args, 'hidden_dim') and args.hidden_dim:
            updated_config['model']['hidden_dim'] = args.hidden_dim
        if hasattr(args, 'dropout') and args.dropout is not None:
            updated_config['model']['dropout'] = args.dropout

        # 训练配置更新
        if hasattr(args, 'epochs') and args.epochs:
            updated_config['training']['num_epochs'] = args.epochs
        if hasattr(args, 'lr') and args.lr:
            updated_config['training']['learning_rate'] = args.lr
        if hasattr(args, 'patience') and args.patience:
            updated_config['training']['early_stopping']['patience'] = args.patience

        # 日志配置更新
        if hasattr(args, 'no_logging') and args.no_logging:
            updated_config['logging']['enable'] = False
        if hasattr(args, 'save_dir') and args.save_dir:
            updated_config['data']['paths']['save_dir'] = args.save_dir

        return updated_config

    def update_config_from_dict(self, config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """从字典更新配置（深度更新）"""

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        updated_config = deepcopy(config)
        deep_update(updated_config, updates)
        return updated_config




@dataclass
class DataConfig:
    """数据配置数据类"""
    csv_path: str = "data/STEAD/chunk2.csv"
    wave_path: str = "data/STEAD/chunk2.hdf5"
    save_dir: str = "results"

    # 过滤参数
    filter_params: Dict[str, Any] = field(default_factory=lambda: {
        'trace_category': 'earthquake_local',
        'source_magnitude_type': 'ml'
    })

    # 预处理参数
    window_samples: int = 500
    target_length: int = 1000
    sampling_rate: int = 100
    channels: int = 3

    # 频谱参数
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 64
    power: float = 2.0


@dataclass
class ModelConfig:
    """模型配置数据类"""
    input_channels: int = 3
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    num_epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 50
    save_path: str = "checkpoints/best_model.pth"

def create_argparser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="震级预测模型训练")

    parser.add_argument('--config', type=str, default='cfg/base.yaml',
                        help='配置文件路径')
    parser.add_argument('--experiment_name', type=str,
                        help='实验名称')
    parser.add_argument('--seed', type=int,
                        help='随机种子')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                        help='计算设备')
    parser.add_argument('--csv_path', type=str,
                        help='CSV数据文件路径')
    parser.add_argument('--wave_path', type=str,
                        help='波形数据文件路径')
    parser.add_argument('--batch_size', type=int,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int,
                        help='数据加载线程数')

    parser.add_argument('--hidden_dim', type=int,
                        help='隐藏层维度')
    parser.add_argument('--dropout', type=float,
                        help='Dropout率')

    parser.add_argument('--epochs', type=int,
                        help='训练轮数')
    parser.add_argument('--lr', type=float,
                        help='学习率')
    parser.add_argument('--patience', type=int,
                        help='早停耐心值')

    parser.add_argument('--no_logging', action='store_true',
                        help='禁用实验日志记录')
    parser.add_argument('--save_dir', type=str,
                        help='结果保存目录')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    return parser


def load_and_merge_config(config_path: str = "../../cfg/base.yaml",
                          args: Optional[argparse.Namespace] = None,
                          updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """加载并合并配置文件、命令行参数和更新字典"""
    loader = ConfigLoader(config_path)
    config = loader.load_config()

    if args is not None:
        config = loader.update_config_from_args(config, args)
        print("[INFO] 已应用命令行参数更新")

    if updates is not None:
        config = loader.update_config_from_dict(config, updates)
        print("[INFO] 已应用字典更新")

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的完整性和正确性"""
    required_sections = ['experiment', 'data', 'model', 'training', 'logging']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的节: {section}")

    csv_path = config['data']['paths']['csv_path']
    wave_path = config['data']['paths']['wave_path']

    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV文件不存在: {csv_path}")
    if not os.path.exists(wave_path):
        print(f"[WARNING] 波形文件不存在: {wave_path}")

    if config['model']['dropout'] < 0 or config['model']['dropout'] > 1:
        raise ValueError("Dropout率必须在0-1之间")

    if config['training']['learning_rate'] <= 0:
        raise ValueError("学习率必须大于0")

    if config['training']['num_epochs'] <= 0:
        raise ValueError("训练轮数必须大于0")

    print("[INFO] 配置验证通过")
    return True


def print_config_summary(config: Dict[str, Any]):
    """打印配置摘要"""

    exp_config = config['experiment']
    print(f"随机种子: {exp_config['random_seed']}")
    print(f"计算设备: {exp_config['device']}")

    data_config = config['data']
    print(f"CSV文件: {data_config['paths']['csv_path']}")
    print(f"波形文件: {data_config['paths']['wave_path']}")

    model_config = config['model']
    print(f"模型隐藏维度: {model_config['hidden_dim']}")
    print(f"Dropout: {model_config['dropout']}")

    train_config = config['training']
    print(f"训练轮数: {train_config['num_epochs']}")
    print(f"学习率: {train_config['learning_rate']}")
    print(f"批次大小: {config['dataloader']['batch_size']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    try:
        config = load_and_merge_config(
            config_path=args.config,
            args=args
        )


        validate_config(config)
        print_config_summary(config)

    except Exception as e:
        print(f"[ERROR] 配置加载失败: {e}")
        sys.exit(1)

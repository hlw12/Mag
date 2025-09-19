#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 15:03
# @Author  : 上头欢乐送、
# @File    : Net.py
# @Software: PyCharm
# 学习新思想，争做新青年


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveSpectralAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.time_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.freq_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.time_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f, t = x.size()
        freq_att = self.freq_pool(x)  # [B, C, 1, T]
        freq_att = self.freq_attention(freq_att)
        time_att = self.time_pool(x)  # [B, C, F, 1]
        time_att = self.time_attention(time_att)
        attended = x * freq_att * time_att
        return attended

class MultiScaleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, padding=5)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class MagnitudeAwarePooling(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.energy_fc = nn.Linear(1, 3)  # 输出3个权重：max, avg, attention

    def forward(self, x):

        B, L, D = x.size()
        energy = torch.mean(x ** 2, dim=(1, 2), keepdim=True)  # [B, 1]
        max_pool = torch.max(x, dim=1)[0]  # [B, D]
        avg_pool = torch.mean(x, dim=1)  # [B, D]
        attn_weights = F.softmax(torch.sum(x ** 2, dim=-1), dim=-1).unsqueeze(-1)  # [B, L, 1]
        attn_pool = torch.sum(x * attn_weights, dim=1)  # [B, D]
        pool_weights = self.energy_fc(energy)  # [B, 1] -> [B, 3]
        pool_weights = F.softmax(pool_weights, dim=-1).squeeze()  # [B, 3]
        final_pool = (pool_weights[:, 0:1] *  max_pool+
                      pool_weights[:, 1:2] * avg_pool +
                      pool_weights[:, 2:3] * attn_pool)
        return final_pool


class EnhancedTimeBranch(nn.Module):
    """
    增强的时域分支
    """

    def __init__(self, in_channels=3, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.multi_scale1 = MultiScaleConvBlock(in_channels, 64, dropout)
        self.multi_scale2 = MultiScaleConvBlock(64, 128, dropout)
        self.multi_scale3 = MultiScaleConvBlock(128, hidden_dim, dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.magnitude_pool = MagnitudeAwarePooling(hidden_dim)

    def forward(self, x):

        out = self.multi_scale1(x)
        out = self.multi_scale2(out)
        out = self.multi_scale3(out)
        out = self.adaptive_pool(out)  # [B, hidden_dim, 128]
        out = out.transpose(1, 2)  # [B, 128, hidden_dim]
        out = self.transformer(out)
        out = self.magnitude_pool(out)
        return out


class EnhancedFreqBranch(nn.Module):
    """
    增强的频域分支
    """

    def __init__(self, in_channels=3, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            AdaptiveSpectralAttention(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            AdaptiveSpectralAttention(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            AdaptiveSpectralAttention(128)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.dropout(out)
        out = self.conv4(out)
        out = torch.flatten(out, 1)  # [B, 256*4*4]

        return out


class CrossModalFusion(nn.Module):
    def __init__(self, time_dim, freq_dim, fusion_dim=512):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, fusion_dim)
        self.freq_proj = nn.Linear(freq_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2)
        )

    def forward(self, time_feat, freq_feat):

        time_proj = self.time_proj(time_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        freq_proj = self.freq_proj(freq_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        time_attended, _ = self.cross_attention(time_proj, freq_proj, freq_proj)
        freq_attended, _ = self.cross_attention(freq_proj, time_proj, time_proj)
        fused_feat = torch.cat([
            time_attended.squeeze(1),
            freq_attended.squeeze(1)
        ], dim=1)
        output = self.fusion_net(fused_feat)
        return output


class MagnitudeRegressionHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class AdvancedMagnitudeNet(nn.Module):
    """
    高级震级预测网络
    结合多尺度时域特征、自适应频域注意力和跨模态融合
    """

    def __init__(self, input_channels=3, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.time_branch = EnhancedTimeBranch(input_channels, hidden_dim, dropout)
        self.freq_branch = EnhancedFreqBranch(input_channels, dropout)
        time_out_dim = hidden_dim
        freq_out_dim = 256 * 4 * 4  # 256 channels * 4 * 4 spatial
        self.cross_fusion = CrossModalFusion(time_out_dim, freq_out_dim, fusion_dim=512)
        self.regression_head = MagnitudeRegressionHead(512 // 2, dropout)
        self.model_info = {
            'name': 'MagnitudeNet',
            'features': [
                'Multi-scale temporal convolution',
                'Adaptive spectral attention',
                'Cross-modal fusion',
                'Magnitude-aware pooling',
                'Transformer encoding'
            ]
        }

    def forward(self, inputs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            wave, spec = inputs
        else:
            wave, spec = inputs
        time_feat = self.time_branch(wave)  # [B, hidden_dim]
        freq_feat = self.freq_branch(spec)  # [B, freq_out_dim]
        fused_feat = self.cross_fusion(time_feat, freq_feat)
        magnitude = self.regression_head(fused_feat)

        return magnitude

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = self.model_info.copy()
        info.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        })
        return info

def create_magnitude_model(**kwargs):
    return AdvancedMagnitudeNet(**kwargs)


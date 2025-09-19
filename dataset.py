#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 14:39
# @Author  : 上头欢乐送、
# @File    : dataset.py
# @Software: PyCharm
# 学习新思想，争做新青年

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def normalization(data):
    min_value = min(data)
    max_value = max(data)
    new_list = []
    for i in data:
        new_list.append((i - min_value) / (max_value - min_value))
    return new_list


class MyDataset(Dataset):
    def __init__(self, csvPath, wavePath, window_size, mode='train', filter_params=None,
                 sampling_by_magnitude=None):
        self.csvSrc = pd.read_csv(csvPath)
        self.csvSrc = self.csvSrc[self.csvSrc['source_magnitude_type'] == 'ml']
        print(f'csv data successfully loaded! Total events in csv file: {len(self.csvSrc)}.')

        if filter_params:
            original_count = len(self.csvSrc)
            for key, value in filter_params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    self.csvSrc = self.csvSrc[(self.csvSrc[key] >= value[0]) & (self.csvSrc[key] <= value[1])]
                else:
                    self.csvSrc = self.csvSrc[self.csvSrc[key] == value]
            print(f'After filtering: {len(self.csvSrc)} events selected from {original_count}.')

        if sampling_by_magnitude:
            sampled_indices = []
            for mag_range, sample_count in sampling_by_magnitude.items():
                group = self.csvSrc[(self.csvSrc['source_magnitude'] >= mag_range[0]) &
                                    (self.csvSrc['source_magnitude'] < mag_range[1])]
                selected = group.sample(n=min(len(group), sample_count), random_state=42)
                sampled_indices.extend(selected.index.tolist())
            self.csvSrc = self.csvSrc.loc[sampled_indices].reset_index(drop=True)
            print(f'After magnitude sampling: {len(self.csvSrc)} samples.')

        trace_names = self.csvSrc['trace_name'].tolist()
        print(f'Found {len(trace_names)} trace names.')
        self.waveSrc = h5py.File(wavePath, 'r')
        print(f'wave data successfully loaded!')

        self.sampleList = []
        self.mode = mode
        # self.window_size = 1.5
        self.window_samples = int(window_size)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=256,
            win_length=256,
            hop_length=64,
            power=2.0  # 返回幅度谱平方
        )

        for idx, trace_name in enumerate(trace_names):
            try:
                rWave = self.waveSrc.get('data/' + str(trace_name))
                if rWave is None:
                    print(f"Warning: Trace {trace_name} not found in wave data, skipping...")
                    continue

                rWave = np.array(rWave)
                event_row = self.csvSrc.iloc[idx]
                p_arrival_sample = int(event_row['p_arrival_sample'])
                magnitude = float(event_row['source_magnitude'])
                start_sample = max(0, p_arrival_sample - self.window_samples)
                end_sample = min(rWave.shape[0], p_arrival_sample + self.window_samples)
                wave_segment = rWave[start_sample:end_sample]
                target_length = 2 * self.window_samples
                if wave_segment.shape[0] < target_length:
                    padding = target_length - wave_segment.shape[0]
                    if wave_segment.ndim == 1:
                        wave_segment = np.pad(wave_segment, (0, padding), 'constant', constant_values=0)
                    else:
                        wave_segment = np.pad(wave_segment, ((0, padding), (0, 0)), 'constant', constant_values=0)
                elif wave_segment.shape[0] > target_length:
                    wave_segment = wave_segment[:target_length]
                wave_tensor = torch.tensor(wave_segment, dtype=torch.float32)
                if wave_tensor.dim() == 2 and wave_tensor.shape[1] == 3:
                    for i in range(3):
                        channel = wave_tensor[:, i]
                        mean = channel.mean()
                        std = channel.std()
                        wave_tensor[:, i] = (channel - mean) / (std + 1e-6)
                elif wave_tensor.dim() == 1:
                    channel = wave_tensor
                    mean = channel.mean()
                    std = channel.std()
                    wave_tensor = (channel - mean) / (std + 1e-6)

                sample = {
                    'wave_data': wave_tensor,
                    'magnitude': magnitude,
                    'p_arrival_sample': p_arrival_sample,
                    'trace_name': trace_name,
                    'trace_category': event_row.get('trace_category', 'unknown'),
                    'source_distance_km': event_row.get('source_distance_km', -1)
                }

                self.sampleList.append(sample)

            except Exception as e:
                print(f"Error processing trace {trace_name}: {str(e)}")
                continue

        print(f"Successfully loaded {len(self.sampleList)} events.")

    def __getitem__(self, index):
        sample = self.sampleList[index]
        wave = sample['wave_data']
        wave_branch_input = wave.transpose(0,1)
        specs = []
        for ch in range(3):
            spec = self.spec_transform(wave[:, ch])  # [freq_bins, time_frames]
            specs.append(spec.unsqueeze(0))  # [1, freq, time]
        spec_tensor = torch.cat(specs, dim=0)  # [3, freq, time]
        spec_branch_input = spec_tensor  # [1, 3, freq, time]
        magnitude = torch.tensor(sample['magnitude'], dtype=torch.float32)

        return (wave_branch_input, spec_branch_input), magnitude

    def __len__(self):
        return len(self.sampleList)

    def getAllsample(self):
        return self.sampleList

    def get_statistics(self):
        magnitudes = [sample['magnitude'] for sample in self.sampleList]
        return {
            'total_samples': len(self.sampleList),
            'magnitude_range': (min(magnitudes), max(magnitudes)),
            'magnitude_mean': np.mean(magnitudes),
            'magnitude_std': np.std(magnitudes)
        }

    def close(self):
        if hasattr(self, 'waveSrc'):
            self.waveSrc.close()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 14:39
# @Author  : 上头欢乐送、
# @File    : dataset.py
# @Software: PyCharm
# 学习新思想，争做新青年
import sys
# import os
# import cfg
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
    def __init__(self, csvPath, wavePath, window_size,
                 filter_params=None, sampling_by_magnitude=None,
                 alternate_csv_paths=None, alternate_wave_paths=None):
        """
        Args:
            csvPath: primary csv path
            wavePath: primary wave hdf5 path
            window_size: window size in samples
            filter_params: dict of filter conditions
            sampling_by_magnitude: dict of {(mag_min, mag_max): sample_count}
            alternate_csv_paths: list of fallback csv paths
            alternate_wave_paths: list of fallback hdf5 paths (paired with alternate_csv_paths)
        """
        print(alternate_csv_paths, alternate_wave_paths)
        self.window_samples = int(window_size)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=256,
            win_length=256,
            hop_length=64,
            power=2.0
        )

        primary_csv = pd.read_csv(csvPath)
        primary_csv = primary_csv[primary_csv['source_magnitude_type'] == 'ml']
        print(f'Primary CSV loaded: {len(primary_csv)} events.')

        if filter_params:
            original_count = len(primary_csv)
            for key, value in filter_params.items():
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    primary_csv = primary_csv[
                        (primary_csv[key] >= value[0]) & (primary_csv[key] <= value[1])]
                else:
                    primary_csv = primary_csv[primary_csv[key] == value]
            print(f'After filtering: {len(primary_csv)} events from {original_count}.')

        if sampling_by_magnitude:
            sampled_per_range = {}

            for mag_range, sample_count in sampling_by_magnitude.items():
                mag_min, mag_max = mag_range
                group = primary_csv[
                    (primary_csv['source_magnitude'] >= mag_min) &
                    (primary_csv['source_magnitude'] < mag_max)
                ]
                unique_events = group['source_id'].unique()
                n_select = min(len(unique_events), sample_count)
                sampled_events = np.random.choice(unique_events, n_select, replace=False)
                sampled_df = group[group['source_id'].isin(sampled_events)].copy()
                sampled_df['_wave_path'] = wavePath

                print(f'  Range {mag_range}: {n_select}/{sample_count} from primary source.')
                remaining = sample_count - n_select
                if remaining > 0 and alternate_csv_paths:
                    already_used_ids = set(sampled_events)
                    for alt_csv_path, alt_wave_path in zip(alternate_csv_paths, alternate_wave_paths):
                        if alt_csv_path == csvPath:
                            continue  # 跳过与主数据源相同的路径
                        if remaining <= 0:
                            break

                        alt_csv = pd.read_csv(alt_csv_path)
                        alt_csv = alt_csv[alt_csv['source_magnitude_type'] == 'ml']

                        if filter_params:
                            for key, value in filter_params.items():
                                if isinstance(value, (tuple, list)) and len(value) == 2:
                                    alt_csv = alt_csv[
                                        (alt_csv[key] >= value[0]) & (alt_csv[key] <= value[1])]
                                else:
                                    alt_csv = alt_csv[alt_csv[key] == value]

                        alt_group = alt_csv[
                            (alt_csv['source_magnitude'] >= mag_min) &
                            (alt_csv['source_magnitude'] < mag_max)
                        ]
                        # 排除已采样的事件ID
                        alt_group = alt_group[~alt_group['source_id'].isin(already_used_ids)]
                        alt_unique = alt_group['source_id'].unique()

                        n_alt = min(len(alt_unique), remaining)
                        if n_alt == 0:
                            continue

                        alt_sampled_events = np.random.choice(alt_unique, n_alt, replace=False)
                        alt_sampled_df = alt_group[
                            alt_group['source_id'].isin(alt_sampled_events)].copy()
                        alt_sampled_df['_wave_path'] = alt_wave_path

                        sampled_df = pd.concat([sampled_df, alt_sampled_df], ignore_index=True)
                        already_used_ids.update(alt_sampled_events)
                        remaining -= n_alt
                        print(f'    Supplemented {n_alt} events from {alt_csv_path}. '
                              f'Still needed: {remaining}.')

                    if remaining > 0:
                        print(f'  Warning: Range {mag_range} still short by {remaining} events '
                              f'after exhausting all alternate sources.')

                sampled_per_range[mag_range] = sampled_df

            self.csvSrc = pd.concat(list(sampled_per_range.values()),
                                    ignore_index=True)
            print(f'Total after sampling: {len(self.csvSrc)} traces '
                  f'from {self.csvSrc["source_id"].nunique()} events.')

        else:
            self.csvSrc = primary_csv.copy()
            self.csvSrc['_wave_path'] = wavePath

        unique_wave_paths = self.csvSrc['_wave_path'].unique().tolist()
        self.waveSrc = {p: h5py.File(p, 'r') for p in unique_wave_paths}
        print(f'Opened {len(self.waveSrc)} HDF5 file(s).')

        self.sampleList = []
        trace_names = self.csvSrc['trace_name'].tolist()

        for idx, trace_name in enumerate(trace_names):
            try:
                event_row  = self.csvSrc.iloc[idx]
                wave_path  = event_row['_wave_path']
                rWave      = self.waveSrc[wave_path].get('data/' + str(trace_name))

                if rWave is None:
                    print(f"Warning: Trace {trace_name} not found in {wave_path}, skipping...")
                    continue

                rWave            = np.array(rWave)
                p_arrival_sample = int(event_row['p_arrival_sample'])
                source_id        = event_row['source_id']
                magnitude        = float(event_row['source_magnitude'])

                start_sample = max(0, p_arrival_sample - self.window_samples)
                end_sample   = min(rWave.shape[0], p_arrival_sample + self.window_samples)
                wave_segment = rWave[start_sample:end_sample]
                target_length = 2 * self.window_samples

                if wave_segment.shape[0] < target_length:
                    padding = target_length - wave_segment.shape[0]
                    if wave_segment.ndim == 1:
                        wave_segment = np.pad(wave_segment, (0, padding),
                                              'constant', constant_values=0)
                    else:
                        wave_segment = np.pad(wave_segment, ((0, padding), (0, 0)),
                                              'constant', constant_values=0)
                elif wave_segment.shape[0] > target_length:
                    wave_segment = wave_segment[:target_length]

                wave_tensor = torch.tensor(wave_segment, dtype=torch.float32)

                if wave_tensor.dim() == 2 and wave_tensor.shape[1] == 3:
                    f_peak = []
                    for i in range(3):
                        channel = wave_tensor[:, i]
                        mean    = channel.mean()
                        std     = channel.std()
                        peak    = torch.max(torch.abs(channel))
                        wave_tensor[:, i] = (channel - mean) / (std + 1e-6)
                        f_peak.extend([
                            np.log10(std.item()  + 1e-6),
                            np.log10(peak.item() + 1e-6)
                        ])
                    f_peak = torch.tensor(f_peak, dtype=torch.float32)

                elif wave_tensor.dim() == 1:
                    channel = wave_tensor
                    mean    = channel.mean()
                    std     = channel.std()
                    peak    = torch.max(torch.abs(channel))
                    wave_tensor = (channel - mean) / (std + 1e-6)
                    f_peak = torch.tensor([
                        np.log10(std.item()  + 1e-6),
                        np.log10(peak.item() + 1e-6)
                    ], dtype=torch.float32)

                sample = {
                    'wave_data':          wave_tensor,
                    'magnitude':          magnitude,
                    'f_peak':             f_peak,
                    'p_arrival_sample':   p_arrival_sample,
                    'source_id':          source_id,
                    'trace_name':         trace_name,
                    'trace_category':     event_row.get('trace_category', 'unknown'),
                    'source_distance_km': event_row.get('source_distance_km', -1),
                    's_arrival_sample':   float(event_row.get('s_arrival_sample', -1)),
                }
                self.sampleList.append(sample)

            except Exception as e:
                print(f"Error processing trace {trace_name}: {str(e)}")
                continue

        print(f"Successfully loaded {len(self.sampleList)} samples.")

    def __getitem__(self, index):
        sample = self.sampleList[index]
        wave   = sample['wave_data']
        f_peak = sample['f_peak']

        wave_branch_input = wave.transpose(0, 1)

        specs = []
        for ch in range(3):
            spec = self.spec_transform(wave[:, ch])
            specs.append(spec.unsqueeze(0))
        spec_branch_input = torch.cat(specs, dim=0)

        magnitude = torch.tensor(sample['magnitude'], dtype=torch.float32)
        return (wave_branch_input, spec_branch_input, f_peak), magnitude

    def __len__(self):
        return len(self.sampleList)

    def getAllsample(self):
        return self.sampleList

    def get_statistics(self):
        magnitudes = [s['magnitude'] for s in self.sampleList]
        return {
            'total_samples':   len(self.sampleList),
            'magnitude_range': (min(magnitudes), max(magnitudes)),
            'magnitude_mean':  np.mean(magnitudes),
            'magnitude_std':   np.std(magnitudes)
        }

    def close(self):
        for f in self.waveSrc.values():
            f.close()
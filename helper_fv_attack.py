#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:07:00 2022
@author: Derek Duan
"""
from typing import List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""add a/v/x attacks for following vehicles"""

def Add_Anomaly_fv(attack_ip_list,
                   T,
                   n):

    freq_type_dict = {}        # key: attack_ip_str, item: <'cont', 'cluster', 'discrete'>
    bias_dict = {}             # key: attack_ip_str, item: bias
    bias_type_dict = {}        # key: attack_ip_str, item: <'const', 'linear', 'sinusoidal'>
    sin_attack_freq_dict = {}  # key: attack_ip_str, item: sin_freqs
    # sin_freq = 0.5
    stealth_mask_continuous_start_dict = {}
    stealth_mask_cluster_discrete_start_dict = {}

### 1
    freq_type_dict['fv1_Vel'] = 'continuous'
    fv1_vel_bias = -5.0
    bias_dict['fv1_Vel'] = fv1_vel_bias
    bias_type_dict['fv1_Vel'] = 'constant'
    fv1_vel_freq = 0.5
    sin_attack_freq_dict['fv1_Vel'] = fv1_vel_freq
    stealth_mask_fv1_vel = 20
    stealth_mask_continuous_start_dict['fv1_Vel'] = stealth_mask_fv1_vel
    stealth_mask_clu_dis_fv1_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv1_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv1_Vel'] = stealth_mask_clu_dis_fv1_vel

    freq_type_dict['fv1_Pos'] = 'continuous'
    fv1_pos_bias = 3
    bias_dict['fv1_Pos'] = fv1_pos_bias
    bias_type_dict['fv1_Pos'] = 'constant'
    fv1_pos_freq = 10.0
    sin_attack_freq_dict['fv1_Pos'] = fv1_pos_freq
    stealth_mask_fv1_pos = 20
    stealth_mask_continuous_start_dict['fv1_Pos'] = stealth_mask_fv1_pos
    stealth_mask_clu_dis_fv1_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv1_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv1_Pos'] = stealth_mask_clu_dis_fv1_pos

### 2
    freq_type_dict['fv2_Vel'] = 'continuous'
    fv2_vel_bias = 5.0
    bias_dict['fv2_Vel'] = fv2_vel_bias
    bias_type_dict['fv2_Vel'] = 'constant'
    fv2_vel_freq = 0.5
    sin_attack_freq_dict['fv2_Vel'] = fv2_vel_freq
    stealth_mask_fv2_vel = 20
    stealth_mask_continuous_start_dict['fv2_Vel'] = stealth_mask_fv2_vel
    stealth_mask_clu_dis_fv2_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv2_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv2_Vel'] = stealth_mask_clu_dis_fv2_vel

    freq_type_dict['fv2_Pos'] = 'continuous'
    fv2_pos_bias = 5.0
    bias_dict['fv2_Pos'] = fv2_pos_bias
    bias_type_dict['fv2_Pos'] = 'constant'
    fv2_pos_freq = 0.5
    sin_attack_freq_dict['fv2_Pos'] = fv2_pos_freq
    stealth_mask_fv2_pos = 20
    stealth_mask_continuous_start_dict['fv2_Pos'] = stealth_mask_fv2_pos
    stealth_mask_clu_dis_fv2_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv2_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv2_Pos'] = stealth_mask_clu_dis_fv2_pos

### 3
    freq_type_dict['fv3_Vel'] = 'continuous'
    fv3_vel_bias = 2.0
    bias_dict['fv3_Vel'] = fv3_vel_bias
    bias_type_dict['fv3_Vel'] = 'constant'
    fv3_vel_freq = 0.5
    sin_attack_freq_dict['fv3_Vel'] = fv3_vel_freq
    stealth_mask_fv3_vel = 20
    stealth_mask_continuous_start_dict['fv3_Vel'] = stealth_mask_fv3_vel
    stealth_mask_clu_dis_fv3_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv3_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv3_Vel'] = stealth_mask_clu_dis_fv3_vel

    freq_type_dict['fv3_Pos'] = 'continuous'
    fv3_pos_bias = 2.0
    bias_dict['fv3_Pos'] = fv3_pos_bias
    bias_type_dict['fv3_Pos'] = 'constant'
    fv3_pos_freq = 0.5
    sin_attack_freq_dict['fv3_Pos'] = fv3_pos_freq
    stealth_mask_fv3_pos = 20
    stealth_mask_continuous_start_dict['fv3_Pos'] = stealth_mask_fv3_pos
    stealth_mask_clu_dis_fv3_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv3_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv3_Pos'] = stealth_mask_clu_dis_fv3_pos

### 4
    freq_type_dict['fv4_Vel'] = 'continuous'
    fv4_vel_bias = 2.0
    bias_dict['fv4_Vel'] = fv4_vel_bias
    bias_type_dict['fv4_Vel'] = 'constant'
    fv4_vel_freq = 0.5
    sin_attack_freq_dict['fv4_Vel'] = fv4_vel_freq
    stealth_mask_fv4_vel = 20
    stealth_mask_continuous_start_dict['fv4_Vel'] = stealth_mask_fv4_vel
    stealth_mask_clu_dis_fv4_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv4_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv4_Vel'] = stealth_mask_clu_dis_fv4_vel

    freq_type_dict['fv4_Pos'] = 'continuous'
    fv4_pos_bias = 2.0
    bias_dict['fv4_Pos'] = fv4_pos_bias
    bias_type_dict['fv4_Pos'] = 'constant'
    fv4_pos_freq = 0.5
    sin_attack_freq_dict['fv4_Pos'] = fv4_pos_freq
    stealth_mask_fv4_pos = 20
    stealth_mask_continuous_start_dict['fv4_Pos'] = stealth_mask_fv4_pos
    stealth_mask_clu_dis_fv4_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv4_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv4_Pos'] = stealth_mask_clu_dis_fv4_pos

### 5
    freq_type_dict['fv5_Vel'] = 'continuous'
    fv5_vel_bias = -1.0
    bias_dict['fv5_Vel'] = fv5_vel_bias
    bias_type_dict['fv5_Vel'] = 'constant'
    fv5_vel_freq = 0.5
    sin_attack_freq_dict['fv5_Vel'] = fv5_vel_freq
    stealth_mask_fv5_vel = 20
    stealth_mask_continuous_start_dict['fv5_Vel'] = stealth_mask_fv5_vel
    stealth_mask_clu_dis_fv5_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv5_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv5_Vel'] = stealth_mask_clu_dis_fv5_vel

    freq_type_dict['fv5_Pos'] = 'continuous'
    fv5_pos_bias = 0.3
    bias_dict['fv5_Pos'] = fv5_pos_bias
    bias_type_dict['fv5_Pos'] = 'constant'
    fv5_pos_freq = 0.5
    sin_attack_freq_dict['fv5_Pos'] = fv5_pos_freq
    stealth_mask_fv5_pos = 20
    stealth_mask_continuous_start_dict['fv5_Pos'] = stealth_mask_fv5_pos
    stealth_mask_clu_dis_fv5_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv5_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv1_Pos'] = stealth_mask_clu_dis_fv5_pos

### 6
    freq_type_dict['fv6_Vel'] = 'continuous'
    fv6_vel_bias = 2.0
    bias_dict['fv6_Vel'] = fv6_vel_bias
    bias_type_dict['fv6_Vel'] = 'constant'
    fv6_vel_freq = 0.5
    sin_attack_freq_dict['fv6_Vel'] = fv6_vel_freq
    stealth_mask_fv6_vel = 20
    stealth_mask_continuous_start_dict['fv6_Vel'] = stealth_mask_fv6_vel
    stealth_mask_clu_dis_fv6_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv6_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv6_Vel'] = stealth_mask_clu_dis_fv6_vel

    freq_type_dict['fv6_Pos'] = 'continuous'
    fv6_pos_bias = 2.0
    bias_dict['fv6_Pos'] = fv6_pos_bias
    bias_type_dict['fv6_Pos'] = 'constant'
    fv6_pos_freq = 0.5
    sin_attack_freq_dict['fv6_Pos'] = fv6_pos_freq
    stealth_mask_fv6_pos = 20
    stealth_mask_continuous_start_dict['fv6_Pos'] = stealth_mask_fv6_pos
    stealth_mask_clu_dis_fv6_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv6_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv6_Pos'] = stealth_mask_clu_dis_fv6_pos

### 7
    freq_type_dict['fv7_Vel'] = 'continuous'
    fv7_vel_bias = 2.0
    bias_dict['fv7_Vel'] = fv7_vel_bias
    bias_type_dict['fv7_Vel'] = 'constant'
    fv7_vel_freq = 0.5
    sin_attack_freq_dict['fv7_Vel'] = fv7_vel_freq
    stealth_mask_fv7_vel = 20
    stealth_mask_continuous_start_dict['fv7_Vel'] = stealth_mask_fv7_vel
    stealth_mask_clu_dis_fv7_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv7_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv7_Vel'] = stealth_mask_clu_dis_fv7_vel

    freq_type_dict['fv7_Pos'] = 'continuous'
    fv7_pos_bias = 2.0
    bias_dict['fv7_Pos'] = fv7_pos_bias
    bias_type_dict['fv7_Pos'] = 'constant'
    fv7_pos_freq = 0.5
    sin_attack_freq_dict['fv7_Pos'] = fv7_pos_freq
    stealth_mask_fv7_pos = 20
    stealth_mask_continuous_start_dict['fv7_Pos'] = stealth_mask_fv7_pos
    stealth_mask_clu_dis_fv7_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv7_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv7_Pos'] = stealth_mask_clu_dis_fv7_pos

### 8
    freq_type_dict['fv8_Vel'] = 'continuous'
    fv8_vel_bias = 2.0
    bias_dict['fv8_Vel'] = fv8_vel_bias
    bias_type_dict['fv8_Vel'] = 'constant'
    fv8_vel_freq = 0.5
    sin_attack_freq_dict['fv8_Vel'] = fv8_vel_freq
    stealth_mask_fv8_vel = 20
    stealth_mask_continuous_start_dict['fv8_Vel'] = stealth_mask_fv8_vel
    stealth_mask_clu_dis_fv8_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv8_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv8_Vel'] = stealth_mask_clu_dis_fv8_vel

    freq_type_dict['fv8_Pos'] = 'continuous'
    fv8_pos_bias = 2.0
    bias_dict['fv8_Pos'] = fv8_pos_bias
    bias_type_dict['fv8_Pos'] = 'constant'
    fv8_pos_freq = 0.5
    sin_attack_freq_dict['fv8_Pos'] = fv8_pos_freq
    stealth_mask_fv8_pos = 20
    stealth_mask_continuous_start_dict['fv8_Pos'] = stealth_mask_fv8_pos
    stealth_mask_clu_dis_fv8_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv8_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv8_Pos'] = stealth_mask_clu_dis_fv8_pos

### 9
    freq_type_dict['fv9_Vel'] = 'continuous'
    fv9_vel_bias = 2.0
    bias_dict['fv9_Vel'] = fv9_vel_bias
    bias_type_dict['fv9_Vel'] = 'constant'
    fv9_vel_freq = 0.5
    sin_attack_freq_dict['fv9_Vel'] = fv9_vel_freq
    stealth_mask_fv9_vel = 20
    stealth_mask_continuous_start_dict['fv9_Vel'] = stealth_mask_fv9_vel
    stealth_mask_clu_dis_fv9_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv9_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv9_Vel'] = stealth_mask_clu_dis_fv9_vel

    freq_type_dict['fv9_Pos'] = 'continuous'
    fv9_pos_bias = 2.0
    bias_dict['fv9_Pos'] = fv9_pos_bias
    bias_type_dict['fv9_Pos'] = 'constant'
    fv9_pos_freq = 0.5
    sin_attack_freq_dict['fv9_Pos'] = fv9_pos_freq
    stealth_mask_fv9_pos = 20
    stealth_mask_continuous_start_dict['fv9_Pos'] = stealth_mask_fv9_pos
    stealth_mask_clu_dis_fv9_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv9_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv9_Pos'] = stealth_mask_clu_dis_fv9_pos

### 10
    freq_type_dict['fv10_Vel'] = 'continuous'
    fv10_vel_bias = 2.0
    bias_dict['fv10_Vel'] = fv10_vel_bias
    bias_type_dict['fv10_Vel'] = 'constant'
    fv10_vel_freq = 0.5
    sin_attack_freq_dict['fv10_Vel'] = fv10_vel_freq
    stealth_mask_fv10_vel = 20
    stealth_mask_continuous_start_dict['fv10_Vel'] = stealth_mask_fv10_vel
    stealth_mask_clu_dis_fv10_vel = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv10_vel.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv10_Vel'] = stealth_mask_clu_dis_fv10_vel

    freq_type_dict['fv10_Pos'] = 'continuous'
    fv10_pos_bias = 2.0
    bias_dict['fv10_Pos'] = fv10_pos_bias
    bias_type_dict['fv10_Pos'] = 'constant'
    fv10_pos_freq = 0.5
    sin_attack_freq_dict['fv10_Pos'] = fv10_pos_freq
    stealth_mask_fv10_pos = 20
    stealth_mask_continuous_start_dict['fv10_Pos'] = stealth_mask_fv10_pos
    stealth_mask_clu_dis_fv10_pos = []
    for i in range(1, 6):
        rand_idx = 10 * (2 * i - 1)
        stealth_mask_clu_dis_fv10_pos.append(rand_idx)

    stealth_mask_cluster_discrete_start_dict['fv10_Pos'] = stealth_mask_clu_dis_fv10_pos

    """add x/v/u attacks for fvs"""
    columnName = []
    for i in range(1, n + 1):
        columnName += [f"fv{i}_Pos"]

    for i in range(1, n + 1):
        columnName += [f"fv{i}_Vel"]

    for i in range(1, n + 1):
        columnName += [f"fv{i}_Acc"]


    fv_stealthmask_df = pd.DataFrame(np.zeros((T,3*n)), columns=columnName)
    fv_attack_df = pd.DataFrame(np.zeros((T,3*n)), columns=columnName)

    for attack_ip_str in attack_ip_list:
        if freq_type_dict[attack_ip_str] == 'continuous':
             fv_stealthmask_df[attack_ip_str].iloc[stealth_mask_continuous_start_dict[attack_ip_str]:] = 1
        elif freq_type_dict[attack_ip_str] == 'cluster':
            window = 10
            for i in range(len(stealth_mask_cluster_discrete_start_dict[attack_ip_str])):
                fv_stealthmask_df[attack_ip_str].iloc[stealth_mask_cluster_discrete_start_dict[attack_ip_str][i]:stealth_mask_cluster_discrete_start_dict[attack_ip_str][i]+window] = 1

        elif freq_type_dict[attack_ip_str] == 'discrete':
            for i in range(len(stealth_mask_cluster_discrete_start_dict[attack_ip_str])):
                fv_stealthmask_df[attack_ip_str].iloc[stealth_mask_cluster_discrete_start_dict[attack_ip_str][i]] = 1

        else:
            print("Unknown freq_type")

    for attack_ip_str in attack_ip_list:
        bias = bias_dict[attack_ip_str]
        if bias_type_dict[attack_ip_str] == 'constant':
            fv_attack_df[attack_ip_str].loc[fv_stealthmask_df[attack_ip_str] == 1] += bias
        elif bias_type_dict[attack_ip_str] == 'linear':
            if freq_type_dict[attack_ip_str] == 'cluster':
                for rand_idx in stealth_mask_cluster_discrete_start_dict[attack_ip_str]:
                    window = 10
                    time_array = np.arange(0, window, 0.1)
                    stealth_length = window
                    fv_attack_df[attack_ip_str].iloc[rand_idx:rand_idx + window] += bias * time_array[:stealth_length]

            elif freq_type_dict[attack_ip_str] == 'continuous':
                time_array = np.arange(0, len(fv_stealthmask_df[attack_ip_str] == 1), 0.1)
                stealth_length = len(fv_attack_df[attack_ip_str].loc[fv_stealthmask_df[attack_ip_str] == 1])
                fv_attack_df[attack_ip_str].loc[fv_stealthmask_df[attack_ip_str] == 1] += bias * time_array[:stealth_length]
            else:
                fv_attack_df[attack_ip_str].loc[fv_stealthmask_df[attack_ip_str] == 1] += bias
        elif bias_type_dict[attack_ip_str] == 'sinusoidal':
            sin_freq = sin_attack_freq_dict[attack_ip_str]
            time_array = np.arange(0, len(fv_stealthmask_df[attack_ip_str] == 1), 0.05)
            sin_bias = bias * np.sin(time_array * sin_freq)
            fv_attack_df[attack_ip_str].loc[fv_stealthmask_df[attack_ip_str] == 1] += sin_bias[:np.count_nonzero(fv_stealthmask_df[attack_ip_str] == 1)]
        else:
            print('Unknown bias_type')

    fv_attack_df_final = Clip_AttackBias_fv(fv_attack_df=fv_attack_df, n=n)

    return fv_attack_df_final, fv_stealthmask_df


def Clip_AttackBias_fv(fv_attack_df,       # limit bias, not the final value
                   n,
                   min_bounds=None,
                   max_bounds=None):
    if min_bounds is None:
        min_bounds = {}
        min_bounds['Acc'] = -8.0
        min_bounds['Vel'] = -50.0
    if max_bounds is None:
        max_bounds = {}
        max_bounds['Acc'] = 8.0
        max_bounds['Vel'] = 50.0

    for i in range(1, n + 1):
        fv_attack_df[f"fv{i}_Acc"] = np.clip(fv_attack_df[f"fv{i}_Acc"].values, min_bounds['Acc'], max_bounds['Acc'])
        fv_attack_df[f"fv{i}_Vel"] = np.clip(fv_attack_df[f"fv{i}_Vel"].values, min_bounds['Vel'], max_bounds['Vel'])

    return fv_attack_df

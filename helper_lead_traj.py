#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:24:46 2021
@author: srivalli
"""
import pandas as pd
import numpy as np
from scipy import stats


# %%

def Data_Cleanup(raw_df):
    raw_df_truncated = raw_df.loc[:, '1':'30']
    column_names = raw_df_truncated.iloc[0]
    column_names_list = column_names.tolist()
    raw_df_truncated.columns = column_names_list
    raw_df_truncated = raw_df_truncated.set_index(pd.Series(np.arange(len(raw_df_truncated))))
    raw_df_truncated = raw_df_truncated.iloc[1:]
    raw_df_truncated = raw_df_truncated.reset_index(drop=True)
    # =============================================================================
    # SimTime
    # LonAccel
    # LatAccel
    # Throttle
    # Brake
    # Gear
    # Heading
    # HeadingError
    # HeadwayDistance
    # HeadwayTime
    # Lane
    # LaneOffset
    # RoadOffset
    # Steer
    # TailwayDistance
    # TailwayTime
    # Velocity
    # LatVelocity
    # VertVelocity
    # XPos
    # YPos
    # ZPos
    # Roll
    # Pitch
    # Yaw
    # EngineRPM
    # SlipFR
    # SlipFL
    # SlipRR
    # SlipRL
    # ============================================================================

    begin = 1000
    end = len(raw_df_truncated.index) - 1000

    preceding_df = raw_df_truncated.loc[begin:end, ['SimTime', 'Velocity', 'LonAccel', 'Steer', 'Lane']]
    preceding_df = preceding_df.astype(float)
    preceding_df = preceding_df.rename(columns={'SimTime': 'SimTime',
                                                'Velocity': 'Prec_Vel',
                                                'LonAccel': 'Prec_Acc',
                                                'Steer': 'Prec_Steer',
                                                'Lane': 'Prec_Lane'})
    preceding_df.drop_duplicates(subset='SimTime',
                                 keep='first',
                                 inplace=True)
    preceding_df['Prec_dT'] = preceding_df['SimTime'].diff().tolist()
    preceding_df['Prec_dT'].iloc[:len(preceding_df) - 1] = preceding_df['Prec_dT'].iloc[1:].values
    preceding_df.reset_index(drop=True, inplace=True)
    preceding_df['Prec_Pos'] = np.zeros(len(preceding_df))
    prec_pos_array = Compute_Prec_Pos(preceding_event=preceding_df,
                                      initial_pos=100.0)
    preceding_df['Prec_Pos'] = prec_pos_array

    preceding_df.reset_index(drop=True,
                             inplace=True)
    return preceding_df


def Gen_Prec_Trunc_Events(dat_str):
    raw_df = pd.read_csv(dat_str,
                         sep="\s+",  # separator whitespace
                         index_col=0,
                         header=None)
    preceding_df = Data_Cleanup(raw_df)
    preceding_events_list = []  # list of event dataframes
    lane_id, count = stats.mode(preceding_df['Prec_Lane'].values)
    event_window = 1000  # 10 secs

    event = pd.DataFrame(columns=['Prec_Vel',
                                  'Prec_Acc',
                                  'Prec_Steer',
                                  'Prec_Lane',
                                  'Prec_dT',
                                  'Prec_Pos'])
    flag = np.zeros(len(preceding_df))
    count = 0

    for cursor in range(len(preceding_df)):
        if (preceding_df['Prec_Lane'].iloc[cursor] == lane_id):
            flag[cursor] = 1
            count += 1
        else:
            if (count >= event_window):
                preceding_event = event.append(preceding_df.loc[flag == 1])
                preceding_event.reset_index(drop=True, inplace=True)
                preceding_events_list.append(preceding_event)
                flag = np.zeros(len(preceding_df))
                event = pd.DataFrame(columns=['Prec_Vel',
                                              'Prec_Acc',
                                              'Prec_Steer',
                                              'Prec_Lane',
                                              'Prec_dT',
                                              'Prec_Pos'])
            count = 0
    return preceding_events_list


def Compute_Prec_Pos(preceding_event, initial_pos):
    prec_pos_array = np.zeros(len(preceding_event))
    kin_array = np.zeros((len(preceding_event), 1))
    ut_array = np.zeros((len(preceding_event), 1))
    at2_array = np.zeros((len(preceding_event), 1))

    # mask = np.ones( (len(preceding_event), len(preceding_event)) , dtype=float ) #### Gives memory error
    # mask = np.triu(mask, k= -1)

    ut_array[:, 0] = np.multiply(preceding_event['Prec_Vel'].values, preceding_event['Prec_dT'].values)
    at2_array[:, 0] = np.multiply(preceding_event['Prec_Acc'].values, np.square(preceding_event['Prec_dT'].values))
    kin_array[:, 0] = ut_array[:, 0] + 0.5 * at2_array[:, 0]
    prec_pos_array[0] = initial_pos

    for i in np.arange(1, len(preceding_event)):
        prec_pos_array[i] = prec_pos_array[i - 1] + kin_array[i - 1]

    return prec_pos_array
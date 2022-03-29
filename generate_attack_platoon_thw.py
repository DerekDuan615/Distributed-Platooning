#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:07:00 2022
@author: Hanyu Zhang, Derek Duan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_naive_platoon_thw import *
from helper_hdv_traj import *
from helper_fv_attack import *

"""Parameter settings"""
n = 10
T = 100
tau = 0.4
L = 3
a_min = -6
a_max = 5
v_min = 0
v_max = 32
# sd = 50
p = 1
delta = 5
init_vel = 20
alpha = [0.6*n**2 - 0.6*(n-i) for i in range(1, n+1)]
beta = [0.4*n**2 - 1.2*(n-i) for i in range(1, n+1)]
Q_x = np.diag(alpha)
Q_v = np.diag(beta)
hessian = [(alpha[i] * (tau ** 2 / 2 + p * tau**2) ** 2 + alpha[i + 1] * (tau**2/2) **2 + tau ** 2 * (1 + beta[i] + beta[i + 1])) for i in range(n - 1)]
hessian.append(alpha[n-1] * (tau ** 2 / 2) ** 2 + tau ** 2 + beta[n-1] * tau ** 2)
KKT = np.diag(hessian)

"""set up hdv/cav and generate platoon instance"""
hdv, hdv_traj = simulate_hdv_trajectory(T, tau, init_vel)

"""Add x/v/u fv attacks"""
fv_attack_df_final, fv_stealthmask_df = Add_Anomaly_fv(attack_ip_list=['fv1_Vel'],T=T,n=n)

"""Set up platoon"""
cav_info, columnName = set_platoon_initial_info(L, p, tau, init_vel, delta, n)
naiveplatoon = NaivePlatoon(n=n,
                       k=0,
                       cav_info=cav_info,
                       hdv_info=hdv_traj,
                       columnName=columnName,
                       KKT=KKT,
                       T=T,
                       alpha_vec=alpha,
                       beta_vec=beta,
                       tau=tau,
                       p=p,
                       delta=delta,
                       fv_attack_df_final=fv_attack_df_final)

"""Run the platoon controller"""
for k in range(0, T-1):
    naiveplatoon.mpc_algorithm(k)
naiveplatoon.trajectory["time_step"] = naiveplatoon.trajectory.index/(1/tau)

"""plot to see performance"""
def naiveplatoon_plot(T,n,L):

    # platoon.trajectory.to_excel("output.xlsx")
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ## Position
    df_position = pd.DataFrame()
    for i in range(n + 1):
        pos_stri = "PosPlatoon_" + str(i)
        if i == 0:
            df_position[pos_stri] = naiveplatoon.trajectory['hdv_Pos']
        else:
            df_position[pos_stri] = naiveplatoon.trajectory[f'fv{i}_Pos']

    df_position = df_position.iloc[:(T - 1), :]

    ## Velocity
    df_velocity = pd.DataFrame()
    for i in range(n + 1):
        stri = "VelPlatoon_" + str(i)
        if i == 0:
            df_velocity[stri] = naiveplatoon.trajectory['hdv_Vel']
        else:
            df_velocity[stri] = naiveplatoon.trajectory[f'fv{i}_Vel']

    df_velocity = df_velocity.iloc[:(T-1), :]

    ## Acceleration
    df_acc = pd.DataFrame()
    for i in range(n + 1):
        stri = "AccPlatoon_" + str(i)
        if i == 0:
            df_acc[stri] = naiveplatoon.trajectory['hdv_Acc']
        else:
            df_acc[stri] = naiveplatoon.trajectory[f'fv{i}_Acc']

    df_acc = df_acc.iloc[:(T-1), :]

    ## SpaceHeadway
    df_spacehead = pd.DataFrame()
    q = list(df_position.columns)
    for i in range(1, df_position.shape[1]):
        stri = "Headspace_" + str(i)
        df_spacehead[stri] = df_position[q[i - 1]] - df_position[q[i]] - L

    df_spacehead = df_spacehead.iloc[:(T-1), :]

    ## TimeHeadway
    df_timehead = pd.DataFrame()
    q = list(df_spacehead.columns)
    r = list(df_velocity.columns)
    for i in range(1, df_velocity.shape[1]):
        stri = "TimeHead_" + str(i)
        # print(df_velocity[r[i]])
        # df_timehead[stri] = df_spacehead[q[i - 1]] / df_velocity[r[i]]
        df_timehead[stri] = np.true_divide(np.array(df_spacehead[q[i - 1]]),(np.array(df_velocity[r[i]]))) # add lower bounds to velocity to avoid divide by 0

    df_timehead = df_timehead.iloc[:(T-1), :]

    """df_position.plot()
    df_velocity.plot()
    df_acc.plot()
    df_spacehead.plot()
    df_timehead.plot()"""

    fig, axes = plt.subplots(5,1)
    ax1, ax2, ax3, ax4, ax5 = axes
    ax1.set_ylabel('Position')
    ax2.set_ylabel('Velocity')
    ax3.set_ylabel('Acceleration')
    ax4.set_ylabel('Space-Headway')
    ax5.set_ylabel('Time-Headway')

    df_position.plot(ax=ax1)
    df_velocity.plot(ax=ax2)
    df_acc.plot(ax=ax3)
    df_spacehead.plot(ax=ax4)
    df_timehead.plot(ax=ax5)

naiveplatoon_plot(T,n,L)

Final_traj = naiveplatoon.trajectory


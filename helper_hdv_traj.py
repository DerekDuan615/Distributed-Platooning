#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:07:00 2022
@author: Hanyu Zhang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Simulated HDV acceleration/deceleration data"""

def simulate_hdv_trajectory(T, tau, init_vel):

    hdv = np.zeros((3, T))  # (x, v, u)  3 rows 50 columns
    for t in range(10):
        hdv[2][t] = 1       # 0-10s, LV acc is 1m/s^2
    for t in range(10, 20):
        hdv[2][t] = 0       # 10-20s, LV acc is 0
    for t in range(20, 30):
        hdv[2][t] = -2      # 20-30s, LV acc is -2
    for t in range(30, 35):
        hdv[2][t] = 3       # 30-35s, LV acc is 3
    for t in range(35, 40):
        hdv[2][t] = -3      # 35-40s, LV acc is -3
    for t in range(40, 50):
        hdv[2][t] = 0       # 40-end, LV acc is 0
    for t in range(50, 60):
        hdv[2][t] = 2
    for t in range(60, 70):
        hdv[2][t] = -2
    for t in range(70, 80):
        hdv[2][t] = 1
    for t in range(80, 90):
        hdv[2][t] = -1
    for t in range(90, T):
        hdv[2][t] = 0

    hdv[1][0] = init_vel  # initial velocity = 20m/s
    """Note the dynamics be: x(t) = x(t-1) + tau*v(t-1) + tau^2/2*u(t)"""
    for t in range(1, T):
        hdv[0][t] = hdv[0][t - 1] + tau * hdv[1][t - 1] + tau ** 2 / 2 * hdv[2][t - 1]
        hdv[1][t] = hdv[1][t - 1] + tau * hdv[2][t - 1]

    hdv_traj = pd.DataFrame({"hdv_Pos": hdv[0], "hdv_Vel": hdv[1], "hdv_Acc": hdv[2]})

    return hdv, hdv_traj



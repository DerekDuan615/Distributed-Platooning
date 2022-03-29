#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:07:00 2022
@author: Hanyu Zhang, Derek Duan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_platoon_initial_info(L, p, tau, init_vel, delta, n):
    global cav_info, columnName, i
    auxilary  = 0 # add a large auxilary variable may help you avoid infeasibility issue
    """platoon initial information data: first is one HDV, then follow n many CAVs"""
    cav_info = [[_ * -(L + p*tau*init_vel + delta + auxilary) for _ in range(1, n + 1)], [init_vel for _ in range(1, n + 1)]]
    columnName = ["hdv_Pos", "hdv_Vel", "hdv_Acc"]  # , "iteration_u", "iteration_lambda"
    for i in range(1, n + 1):
        columnName += [f"fv{i}_Pos", f"fv{i}_s", f"fv{i}_Vel", f"fv{i}_Acc", f"fv{i}_h", f"fv{i}_u_ite", f"fv{i}_l_ite"]

    return cav_info, columnName

class NaivePlatoon:
    """platoon object"""
    def __init__(self,
                 n,
                 k,
                 cav_info,
                 hdv_info,
                 columnName,
                 KKT,
                 T,
                 alpha_vec,
                 beta_vec,
                 tau,
                 p,
                 delta,
                 fv_attack_df_final,
                 a_min=-6,
                 a_max=5,
                 v_min=0,
                 v_max=32,
                 L=3,                # constant related to vehicle length
                 ite_max_u=100,
                 ite_max_lambda=100):
        """constructor for the Platoon class"""
        self.hdv = hdv_info
        self.n = n                      # platoon CAV number
        self.N = self.n + 1             # platoon vehicle number
        self.x = np.array(cav_info[0])  # platoon MPC state: location at current step k
        self.v = np.array(cav_info[1])  # platoon MPC state: velocity at current step k
        self.u = np.zeros(n)            # platoon MPC input: acceleration at current step k
        self.KKT = KKT
        self.a_min = a_min
        self.a_max = a_max
        self.v_min = v_min
        self.v_max = v_max
        self.tau = tau
        self.p = p
        self.delta = delta
        self.L = L
        self.T = T
        self.alpha_vec = alpha_vec
        self.beta_vec = beta_vec
        self.ite_max_u = ite_max_u
        self.ite_max_l = ite_max_lambda
        self.fv_attack_df_final = fv_attack_df_final
        """call initialize_control_variable function to initialize iteration variables 
            and initialize trajectory DataFrame"""
        self.initialize_control_variables(k)
        self.trajectory = pd.DataFrame(columns=columnName)  #output dataframe

    def initialize_control_variables(self, k):
        """initialize iteration_varibles for distributed optimization"""
        # self.k = k  # current time step
        self.x_hdv = self.hdv.loc[k, 'hdv_Pos']
        self.v_hdv = self.hdv.loc[k, 'hdv_Vel']
        self.u_hdv = self.hdv.loc[k, 'hdv_Acc']
        self.x_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of x at next step k+1
        self.v_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of v at next step k+1
        self.sd_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of desired spacing at next step k+1

        self.u_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of u at next step k+1
        self.l_ite = np.zeros((self.ite_max_l, self.n))  # iteration of lambda at next step k+1
        self.zx_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of z, spacing error at next step k+1
        self.zv_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of z', velocity error at next step k+1
        self.gradient_ite = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of gradient
        self.u_ite_dir = np.zeros((self.ite_max_l, self.ite_max_u, self.n))  # iteration of u improvement direction

        self.ite_u = 0  # current iteration index of u
        self.ite_l = 0  # current iteration index of lambda
        self.primal_idx = self.ite_u
        self.dual_idx = self.ite_l
        self.primal_stop = False
        self.dual_stop = False

    def mpc_algorithm(self, k):
        """start mpc control for current time step"""

        while self.ite_l < self.ite_max_l - 1 and not self.dual_stop:
            # reset primal information
            self.ite_u = 0
            self.primal_stop = False

            # initialize u iteration information
            self.calculate_ite_state()

            # print(f"ite_lambda:{self.ite_l}")
            # self.ite_u = 0                         # edited by D.Duan
            # self.calculate_ite_state()             # edited by D.Duan
            while self.ite_u < self.ite_max_u - 1 and not self.primal_stop:
                # print(f"ite_u:{self.ite_u}")
                self.compute_primal()
                self.primal_stop = self.check_primal_stop()

            self.compute_dual()
            self.dual_stop = self.check_dual_stop()
            self.primal_idx = self.ite_u

        self.dual_idx = self.ite_l -1
        self.update_state(k)

    def compute_primal(self):  
        """compute and store the primal update for the distributed optimization algorithm"""
        self.ite_u += 1
        self.compute_gradient()
        self.u_ite_dir[self.ite_l][self.ite_u] = - np.dot(np.linalg.inv(self.KKT), self.gradient_ite[self.ite_l][self.ite_u]) # np.linalg.inv(KKT)
        self.u_ite[self.ite_l][self.ite_u] = self.u_ite[self.ite_l][self.ite_u - 1] + \
                                             0.5 * self.u_ite_dir[self.ite_l][self.ite_u]
        for i in range(self.n):
            self.u_ite[self.ite_l][self.ite_u][i] = min(max(self.a_min, self.u_ite[self.ite_l][self.ite_u][i]), self.a_max)
            self.u_ite[self.ite_l][self.ite_u][i] = min(max((self.v_min - self.v_ite[self.ite_l][self.ite_u-1][i]) / self.tau,
                                                            self.u_ite[self.ite_l][self.ite_u][i]),
                                                        (self.v_max - self.v_ite[self.ite_l][self.ite_u-1][i]) / self.tau)

        self.calculate_ite_state()

    def calculate_ite_state(self):
        """calculate iteration state of the CAVs"""
        self.v_ite[self.ite_l][self.ite_u] = self.v[:] + self.tau * self.u_ite[self.ite_l][self.ite_u]  # cavs' next velocity
        self.x_ite[self.ite_l][self.ite_u] = self.x[:] + self.tau * self.v[:] + self.tau**2 /2 * self.u_ite[self.ite_l][self.ite_u] # cavs' next location
        self.zv_ite[self.ite_l][self.ite_u][0] = self.v_hdv + self.tau * self.u_hdv - self.v_ite[self.ite_l][self.ite_u][0] # hdv - first CAV
        self.zx_ite[self.ite_l][self.ite_u][0] = self.x_hdv + self.tau * self.v_hdv + self.tau**2/2 * self.u_hdv - self.x_ite[self.ite_l][self.ite_u][0] -\
                                                 (self.L + self.p*self.tau*self.v_ite[self.ite_l][self.ite_u][0] + self.delta)
        for i in range(self.n - 1):
            self.zv_ite[self.ite_l][self.ite_u][i+1] = self.v_ite[self.ite_l][self.ite_u][i] - \
                                                     self.v_ite[self.ite_l][self.ite_u][i+1]
            self.zx_ite[self.ite_l][self.ite_u][i+1] = self.x_ite[self.ite_l][self.ite_u][i] - \
                                                     self.x_ite[self.ite_l][self.ite_u][i+1] - \
                                                       (self.L + self.p*self.tau*self.v_ite[self.ite_l][self.ite_u][i+1] + self.delta)

    def compute_gradient(self):
        for i in range(self.n - 1):
            self.gradient_ite[self.ite_l][self.ite_u][i] = -self.tau**2 * (1/2 + self.p) * (self.zx_ite[self.ite_l][self.ite_u-1][i]
                                                           *self.alpha_vec[i]) + self.tau**2 / 2 * self.zx_ite[self.ite_l][self.ite_u-1][i+1]*self.alpha_vec[i+1] + \
                                                           self.tau**2 * self.u_ite[self.ite_l][self.ite_u-1][i] + \
                                                           self.tau * (-self.zv_ite[self.ite_l][self.ite_u-1][i]*self.beta_vec[i] +
                                                                  self.zv_ite[self.ite_l][self.ite_u-1][i+1]*self.beta_vec[i+1]) + \
                                                            self.l_ite[self.ite_l][i] * (self.p * self.tau ** 2 + self.tau**2 / 2) - \
                                                            self.l_ite[self.ite_l][i+1] * self.tau**2 / 2


        self.gradient_ite[self.ite_l][self.ite_u][self.n-1] = -self.tau**2 * (1/2 + self.p) * self.zx_ite[self.ite_l][self.ite_u-1][self.n-1] * self.alpha_vec[self.n-1]+\
                                                          self.tau**2 * self.u_ite[self.ite_l][self.ite_u-1][self.n-1] - \
                                                         self.tau * self.zv_ite[self.ite_l][self.ite_u-1][self.n-1]*self.beta_vec[self.n-1] + \
                                                          self.l_ite[self.ite_l][self.n-1] * (self.p * self.tau ** 2 + self.tau ** 2 / 2)

    def check_primal_stop(self):
        """check if the primal can stop or not"""
        for i in range(self.n):
            if abs(self.u_ite[self.ite_l][self.ite_u][i] - self.u_ite[self.ite_l][self.ite_u - 1][i]) > 0.01:
                return False   # continue
        return True            # stop

    def compute_dual(self):
        """compute and store the dual update for the distributed optimization algorithm"""
        self.ite_l += 1

        for i in range(1, self.n):
            a = 0.05
            b = 0.01
            gap =  self.L + self.p * self.tau * self.v_ite[self.ite_l-1][self.ite_u][i] \
                   - (self.x_ite[self.ite_l-1][self.ite_u][i-1] - self.x_ite[self.ite_l-1][self.ite_u][i])

            self.l_ite[self.ite_l][i] = self.l_ite[self.ite_l - 1][i] + 1 / (a + b * self.ite_l ) * max(gap, 0)

    def check_dual_stop(self):    # when lambda value changes a lot: False;  Otherwise: True→Stop
        """check if dual algorithm can stop or not"""
        # if it reaches maximum lambda iteration, it is likely that your attack makes the optimization problem infeasible
        # Return true and stop
        if self.ite_l == self.ite_max_l-2:
            print("Dual Reaches Saturation!!!")
            return True
        for i in range(1, self.n):
            if abs(self.l_ite[self.ite_l - 1][i] - self.l_ite[self.ite_l][i]) > 1:
                    # self.x_ite[self.ite_l][self.ite_u][i-1] - self.x_ite[self.ite_l][self.ite_u][i] \
                    # < self.L + self.p * self.tau * self.v_ite[self.ite_l][self.ite_u][i]:
                return False
        print("No Dual Saturation Stop")
        return True

    def update_state(self, k):
        """update the MPC state to next time step and record trajectory at the same time"""
        self.u = self.u_ite[self.dual_idx][self.primal_idx]
        self.record_trajectory(k)

        self.x = self.x_ite[self.dual_idx][self.primal_idx] + self.fv_attack_df_final.iloc[k].values[0:self.n]         # prev platoon MPC state: location
        self.v = self.v_ite[self.dual_idx][self.primal_idx] + self.fv_attack_df_final.iloc[k].values[self.n:2*self.n]  # prev platoon MPC state: velocity
        self.u = np.zeros(self.n)  # prev platoon MPC input: acceleration
        self.initialize_control_variables(k+1)

    def record_trajectory(self, k):
        """record trajectory to the pandas dataframe"""
        cur_df = {}
        cur_df["hdv_Pos"] = self.hdv.loc[k, 'hdv_Pos']
        cur_df["hdv_Vel"] = self.hdv.loc[k, 'hdv_Vel']
        cur_df["hdv_Acc"] = self.hdv.loc[k, 'hdv_Acc']
        # cur_df["iteration_u"] = self.u_ite
        # cur_df["iteration_lambda"] = self.l_ite

        u_ite_record = [[] for _ in range(self.n)]
        for idx in range(self.primal_idx + 1):
            for i in range(self.n):
                u_ite_record[i].append(self.u_ite[self.dual_idx][idx][i])

        for i in range(self.n):
            cur_df[f"fv{i+1}_Acc"] = self.u[i]
            cur_df[f"fv{i+1}_Vel"] = self.v[i]
            cur_df[f"fv{i+1}_Pos"] = self.x[i]
            cur_df[f"fv{i+1}_l_ite"] = self.l_ite[self.ite_l-1][i]
            cur_df[f"fv{i+1}_u_ite"] = u_ite_record[i]

        cur_df[f"fv1_s"] = self.hdv.loc[k,'hdv_Pos'] - self.x[0] - self.L
        cur_df[f"fv1_h"] = cur_df[f"fv1_s"] / cur_df[f"fv1_Vel"]
        for i in range(1, self.n):
            cur_df[f"fv{i+1}_s"] = self.x[i-1] - self.x[i] - self.L
            cur_df[f"fv{i+1}_h"] = (cur_df[f"fv{i+1}_s"]) / max(cur_df[f"fv{i+1}_Vel"], 0.01) # avoid divide by zero, numerical issue

        self.trajectory = self.trajectory.append(cur_df, ignore_index=True)

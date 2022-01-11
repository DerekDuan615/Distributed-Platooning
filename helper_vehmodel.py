"""
Created on Mon Oct 18 16:30:00 2021

@author: Derek Duan
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# %% Vehicle model
class Veh:

    # Define Communication Buffers
    # Vel_buffer = None
    # Pos_buffer = None
    # Iteration_buffer = None


    def __init__(self,
                 ID,                   # how should I input the value of ID?
                 num_veh,
                 min_speed=0.5,      # m/s
                 max_speed=33.33,    # m/s
                 min_acc=-8,         # m/s^2
                 max_acc=1.35,       # m/s^2
                 gap_l=10,           # constant gap depending on vehicle length (m)
                 delta=50,           # desired constant distance between two adjacent vehicles
                 r=0.2,              # constant reaction time (r>=tau)
                 ):

        self.ID = ID
        self.num_veh = num_veh
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.gap_L = gap_l
        self.delta = delta
        self.r = r

        return


    def Initialize_Variables(self, lead_traj):

        self.lead_traj = lead_traj

        self.ego_traj = pd.DataFrame(columns=['Lead_Vel_Rep',
                                              'Lead_Pos_Rep',
                                              'Lead_Acc_Rep',
                                              'Ego_Vel',
                                              'Ego_Pos',
                                              'Ego_Acc',
                                              'Iteration_u',
                                              'Iteration_lambda'])

        self.ego_traj['Lead_Vel_Rep'] = self.lead_traj['Prec_Vel'].values
        self.ego_traj['Lead_Pos_Rep'] = self.lead_traj['Prec_Pos'].values
        self.ego_traj['Lead_Acc_Rep'] = self.lead_traj['Prec_Acc'].values

        self.lead_vel_array = self.lead_traj['Prec_Vel'].values
        self.lead_pos_array = self.lead_traj['Prec_Pos'].values
        self.lead_acc_array = self.lead_traj['Prec_Acc'].values
        self.dT_array = self.lead_traj['Prec_dT'].values

        self.ego_acc_array = np.zeros(len(self.lead_traj))     # Built for moving this array to the traj table, and status reading
        self.ego_vel_array = np.zeros(len(self.lead_traj))
        self.ego_pos_array = np.zeros(len(self.lead_traj))
        self.ego_iteration_u_list = [[] for x in range(len(self.lead_traj))]  # list not array
        self.ego_iteration_lambda_list = [[] for x in range(len(self.lead_traj))]

        self.ego_pos_array[0] = self.lead_pos_array[0] - (self.delta+2)*self.ID  #
        self.ego_vel_array[0] = self.lead_vel_array[0]

        return

    def Read_Cur_State(self, idx):

        self.ego_vel = self.ego_vel_array[idx]
        self.ego_pos = self.ego_pos_array[idx]
        self.dT = self.dT_array[idx]
        self.idx = idx               # Tell the vehicle which control step it is now

        return

    def Vel_Buffer_Out(self):       # Let the main.py use the current velocity to update the velocity buffer

        cur_vel = self.ego_vel

        return cur_vel

    def Pos_Buffer_Out(self):  # Let the main.py use the current position to update the position buffer

        cur_pos = self.ego_pos

        return cur_pos

    def Initialize_Controller(self, tau, idx, S, Q_alpha, Q_beta, Pos_buffer, Vel_buffer):

        self.z_1 = np.zeros(self.num_veh)
        self.z_2 = np.zeros(self.num_veh)
        for i in range(self.num_veh):
            self.z_1[i] = Pos_buffer[i]-Pos_buffer[i+1]-self.delta
            self.z_2[i] = Vel_buffer[i]-Vel_buffer[i+1]

        self.z = np.concatenate((self.z_1,self.z_2))
        self.c = (self.lead_acc_array[idx]*np.ones(self.num_veh))@np.transpose(np.linalg.inv(S))@(0.25*pow(tau, 4)*Q_alpha+pow(tau, 2)*Q_beta)@np.linalg.inv(S)\
            - self.z@np.vstack(((0.5*pow(tau,2)*Q_alpha),(0.5*pow(tau,3)*Q_alpha+tau*Q_beta)))@np.linalg.inv(S)   # row vector
        # print(self.c)

        # define Box Constraint of u
        self.box_left = max(self.min_acc, (self.min_speed-self.ego_vel)/tau)
        self.box_right = min(self.max_acc, (self.max_speed-self.ego_vel)/tau)

        # Randomly pick initial values of u in the box constraint
        self.u_old = self.min_acc - (self.max_acc - self.min_acc) - 2        # Define a small enough value of u for the 1st iteration
        self.u_cur = random.uniform(0, self.box_right)


        # Update the first element in the iter_u list in the kth control step
        self.counter = 0
        self.ego_iteration_u_list[self.idx].append([])
        # print(self.ego_iteration_u_list)
        self.ego_iteration_u_list[self.idx][self.counter].append(self.u_cur)
        # print(self.ego_iteration_u_list)

        # Get hi and hi+1 (needs current velocity), function gi (updated after Compute_Next_State)
        self.h_1 = self.function_h(Vel_buffer[self.ID], tau)  # hi
        self.h_2 = self.function_h(Vel_buffer[min(self.ID + 1, self.num_veh)], tau)  # hi+1

        return

    def Iteration_Buffer_Out(self):  # Let the main.py use the current u to update the iteration buffer

        cur_iter_u = self.u_cur
        '''print("============================")
        print(type(cur_iter_u))
        print(cur_iter_u)'''

        return cur_iter_u

    def Initialize_Lambda(self, Pos_buffer, Vel_buffer, Iteration_buffer, H, tau):

        inter_u = np.zeros(self.num_veh)

        # print(self.c@np.linalg.inv(H)@np.transpose(self.c))
        # print(self.function_g(tau, Pos_buffer[self.ID-1], Pos_buffer[self.ID], Vel_buffer[self.ID-1], Vel_buffer[self.ID], Iteration_buffer[self.ID-1], Iteration_buffer[self.ID]))
        self.box_lambda = (0.5*inter_u@H@np.transpose(inter_u) + np.dot(self.c, np.transpose(inter_u))\
                      + 0.5*self.c@np.linalg.inv(H)@np.transpose(self.c))/self.function_g(tau, Pos_buffer[self.ID-1], Pos_buffer[self.ID], Vel_buffer[self.ID-1], Vel_buffer[self.ID], Iteration_buffer[self.ID-1], Iteration_buffer[self.ID])

        # Randomly pick initial values of lambda in the box constraint
        self.lambda_new = random.uniform(0, self.box_lambda)

        # Update the first element in the iter_lambda list in the kth control step
        self.ego_iteration_lambda_list[self.idx].append(self.lambda_new)

        eig_val_H, eig_vec_H = np.linalg.eig(H)
        self.miu_min = min(eig_val_H)
        self.Mg = self.function_mg(Vel_buffer, tau)

        return

    def Lambda_Buffer_Out(self):  # Let the main.py use the initial λ

        ini_lambda = self.lambda_new

        return ini_lambda

    def function_g(self,tau,x1,x2,v1,v2,u1,u2):

        output_g = -pow(tau,2)*pow(u2, 2)/(2*self.min_acc) +\
                   (self.r*tau+pow(tau,2)/2-(v2-self.min_speed)*tau/(self.min_acc))*u2\
                   + (self.gap_L-pow(v2-self.min_speed,2)/(2*self.min_acc)+(self.r+tau)*v2-tau*v1+x2-x1-pow(tau,2)*u1/2)

        return output_g

    def function_h(self, v2, tau):

        output_h = self.r*tau+pow(tau,2)/2-(v2-self.min_speed)*tau/(self.min_acc)

        return output_h

    def function_mg(self, Vel_buffer, tau):

        w = np.zeros(self.num_veh)
        h = np.zeros(self.num_veh)
        for i in range(1, self.num_veh+1):
            w[i-1] = min(self.max_acc, (self.max_speed-Vel_buffer[i])/tau)
            h[i-1] = self.function_h(Vel_buffer[i], tau)

        mid_Mg = pow((-pow(tau, 2)/self.min_acc)*w+h, 2)
        Mg = pow(sum(mid_Mg)/len(mid_Mg), 0.5)

        return Mg


    def Compute_Primal(self, tau, H, Iteration_buffer, Lambda_buffer):

        self.epsilon = 0.1   # Choose a small ε>0
        self.theta = 2*self.miu_min/(pow(self.Mg, 2)+2*self.epsilon*self.miu_min)
        eig_val_L, eig_vec_L = np.linalg.eig(H+(-pow(tau, 2)/self.min_acc)*np.diag(Lambda_buffer))
        self.L = max(eig_val_L)
        sai = random.uniform(0.000001, 0.2/self.L)

        sigma_1 = 0.01  # threshold |u_new-u_old|

        mid_value = np.dot(H,np.transpose(Iteration_buffer[1:self.num_veh+1]))[self.ID-1] + self.c[self.ID-1] - pow(tau, 2)*self.u_cur/self.min_acc + self.h_1 + self.h_2
        u_new = self.Box(self.u_cur-sai*mid_value, self.box_left, self.box_right)
        self.u_old = self.u_cur
        self.u_cur = u_new
        self.ego_iteration_u_list[self.idx][self.counter].append(self.u_cur)

        if abs(self.u_cur-self.u_old) < sigma_1:
            flag_u = 1
        else:
            flag_u = 0

        return flag_u


    def Compute_Dual(self, tau, Pos_buffer, Vel_buffer, Iteration_buffer):

        sigma_2 = 0.01     # threshold |λ_new-λ_old|
        lambda_old = self.lambda_new
        mid_value = self.function_g(tau, Pos_buffer[self.ID-1], Pos_buffer[self.ID], Vel_buffer[self.ID-1], Vel_buffer[self.ID], Iteration_buffer[self.ID-1], Iteration_buffer[self.ID])
        self.lambda_new = self.Box(lambda_old + self.theta*(mid_value - self.epsilon*lambda_old), 0, self.box_lambda)
        self.ego_iteration_lambda_list[self.idx].append(self.lambda_new)
        if abs(self.lambda_new - lambda_old) < sigma_2:
            flag_lambda = 1
        else:
            flag_lambda = 0

        self.counter = self.counter + 1
        self.ego_iteration_u_list[self.idx].append([])

        return flag_lambda

    def Pop_u_list(self):

        self.ego_iteration_u_list[self.idx].pop()

        return

    def Box(self,input,low_bound, up_bound):

        if input <= low_bound:
            output = low_bound
        elif input >= up_bound:
            output = up_bound
        else:
            output = input

        return output

    def Compute_Next_State(self):

        self.next_ego_vel = self.ego_vel + self.u_cur * self.dT
        s_ego = self.ego_vel * self.dT + 0.5 * self.u_cur * (self.dT ** 2)
        self.next_ego_pos = self.ego_pos + s_ego

        return


    def Update_Ego_State(self):

        self.ego_acc_array[self.idx] = self.u_cur
        self.ego_vel_array[self.idx + 1] = self.next_ego_vel
        self.ego_pos_array[self.idx + 1] = self.next_ego_pos

        return


    def Update_Ego_Traj_Df(self):

        self.ego_traj['Ego_Vel'] = self.ego_vel_array
        self.ego_traj['Ego_Pos'] = self.ego_pos_array
        self.ego_traj['Ego_Acc'] = self.ego_acc_array
        self.ego_traj['Iteration_u'] = self.ego_iteration_u_list
        self.ego_traj['Iteration_lambda'] = self.ego_iteration_lambda_list

        # print(self.ego_iteration_u_list)

        return
















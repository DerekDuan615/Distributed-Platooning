"""
Created on Mon Oct 27 14:38:00 2021

@author: Derek Duan
"""
# %%
import numpy as np
from helper_vehmodel import *   # set the folder path to folder containing these two files
from helper_lead_traj import *
import matplotlib.pyplot as plt
import pandas as pd

# %%  Global Parameters
num_veh = 5
tau = 0.1    # control step sampling time
alpha = 2    # penalty parameter alpha (assume all vehicles have the same value)
beta = 1     # penalty parameter beta (assume all vehicles have the same value)

D_alpha = np.diag([alpha]*num_veh)           # diagonal matrix of penalty weight alpha
D_beta = np.diag([beta]*num_veh)

E = np.ones((num_veh, num_veh))
S = np.tril(E)                               # Genarate the lower triangular matrix S in the paper
M = np.dot(np.transpose(S), S)               # Calculate transpose(S)*S in the paper
eig_value, eig_vectors = np.linalg.eig(M)    # Calculate the eigenvalues and eigenvectors
P = np.transpose(eig_vectors)                # Orthogonal matrix P in the paper
Q_alpha = np.transpose(P)@D_alpha@P          # Penalty matrix Qz  in the paper
Q_beta = np.transpose(P)@D_beta@P            # Penalty matrix Qz' in the paper

Q_alpha_1 = Q_alpha[0, :]
Q_alpha_2 = Q_alpha[1:num_veh, :]
Q_alpha_3 = Q_alpha[0, 0]
Q_alpha_4 = Q_alpha[0, 1:num_veh]
Q_alpha_5 = Q_alpha[1:num_veh, 1:num_veh]
Q_alpha_6 = Q_alpha[:, 0]
Q_alpha_7 = Q_alpha[:, 1:num_veh]

Q_beta_1 = Q_beta[0, :]
Q_beta_2 = Q_beta[1:num_veh, :]
Q_beta_3 = Q_beta[0, 0]
Q_beta_4 = Q_beta[0, 1:num_veh]
Q_beta_5 = Q_beta[1:num_veh, 1:num_veh]
Q_beta_6 = Q_beta[:, 0]
Q_beta_7 = Q_beta[:, 1:num_veh]

Q_alpha_2_new = np.row_stack((Q_alpha_2, np.zeros(num_veh)))
Q_alpha_5_new = np.column_stack((np.row_stack((Q_alpha_5, np.zeros(num_veh-1))), np.zeros(num_veh)))

Q_beta_2_new = np.row_stack((Q_beta_2, np.zeros(num_veh)))
Q_beta_5_new = np.column_stack((np.row_stack((Q_beta_5, np.zeros(num_veh-1))), np.zeros(num_veh)))

H = pow(tau, 4)/4*(Q_alpha-2*Q_alpha_2_new+Q_alpha_5_new) + pow(tau, 2)*(Q_beta-2*Q_beta_2_new+Q_beta_5_new)\
    + pow(tau, 2)*np.eye(num_veh)   # Calculate the matrix of H

Q_alpha_4_new = np.pad(Q_alpha_4, (0, 1))
Q_alpha_7_new = np.column_stack((Q_alpha_7, np.zeros(num_veh)))

Q_beta_4_new = np.pad(Q_beta_4, (0, 1))
Q_beta_7_new = np.column_stack((Q_beta_7, np.zeros(num_veh)))

# %% Get the preceding vehicle's trajectory
# parent_path = '/home/srivalli/Desktop/Raccon_Sensor_Ext/'
dat_str = 'D:\PycharmProjects\PythonProjects\DistributedPlatooning\Experimenter_0330_caccHwDenDayWindy_1553972506.dat'
raw_df = pd.read_csv(dat_str,
                 sep="\s+", #separator whitespace
                 index_col=0,
                 header=None)

lead_traj = Data_Cleanup(raw_df=raw_df)
# %%

start = len(lead_traj)-70000
end = start + 30000
lead_traj = lead_traj.iloc[start:end]
lead_traj.reset_index(drop=True, inplace=True)

lead_vel_array = lead_traj['Prec_Vel'].values
lead_pos_array = lead_traj['Prec_Pos'].values
lead_acc_array = lead_traj['Prec_Acc'].values

# %% Generate following vehicle objects in the platoon
Veh_obj_dict = {}

for veh_id in range(1, num_veh+1):

    veh_obj = Veh(ID=veh_id,
                  num_veh=num_veh)
    Veh_obj_dict[veh_id] = veh_obj

# Define Communication Buffers
Vel_buffer = np.zeros(num_veh + 1)
Pos_buffer = np.zeros(num_veh + 1)
Iteration_buffer = np.zeros(num_veh + 1)
Lambda_buffer = np.zeros(num_veh)
# Veh.Vel_buffer = np.zeros(num_veh + 1)

# %% Generate platoon trajectories
for veh_id in range(1, num_veh+1):
    Veh_obj_dict[veh_id].Initialize_Variables(lead_traj)

for k in range(len(lead_traj)-1):

    control_step = k
    for veh_id in range(1, num_veh + 1):
        Veh_obj_dict[veh_id].Read_Cur_State(control_step)

    Vel_buffer[0] = lead_vel_array[k]        # The first value of all buffers is the value from preceding vehicle at time k
    Pos_buffer[0] = lead_pos_array[k]
    Iteration_buffer[0] = lead_acc_array[k]

    for veh_id in range(1, num_veh+1):
        Vel_buffer[veh_id] = Veh_obj_dict[veh_id].Vel_Buffer_Out()  # Update Velocity Buffer
        Pos_buffer[veh_id] = Veh_obj_dict[veh_id].Pos_Buffer_Out()  # Update Position Buffer

    for veh_id in range(1, num_veh+1):
        Veh_obj_dict[veh_id].Initialize_Controller(tau, Q_alpha_4_new, Q_alpha_1, Q_alpha_7_new, Q_alpha, Q_beta_4_new,
                                                   Q_beta_1, Q_beta_7_new, Q_beta, Vel_buffer, Pos_buffer)

    for veh_id in range(1, num_veh+1):
        Iteration_buffer[veh_id] = Veh_obj_dict[veh_id].Iteration_Buffer_Out()  # Update Iteration Buffer

    for veh_id in range(1, num_veh + 1):
        Veh_obj_dict[veh_id].Initialize_Lambda(Pos_buffer, Vel_buffer, Iteration_buffer, H, tau)

    for veh_id in range(0, num_veh):
        Lambda_buffer[veh_id] = Veh_obj_dict[veh_id+1].Lambda_Buffer_Out()  # Update Lambda Buffer


    sum_flag_lambda = 0
    while sum_flag_lambda != num_veh:
        flag_lambda = np.zeros(num_veh)
        sum_flag_u = 0
        flag_u = np.zeros(num_veh)
        while sum_flag_u != num_veh:
            for veh_id in range(1, num_veh+1):
                Iteration_buffer[veh_id] = Veh_obj_dict[veh_id].Iteration_Buffer_Out()  # Update Iteration Buffer

            for veh_id in range(1, num_veh + 1):
                flag_u[veh_id-1] = Veh_obj_dict[veh_id].Compute_Primal(tau, H, Iteration_buffer, Lambda_buffer)

            sum_flag_u = sum(flag_u)

        for veh_id in range(1, num_veh + 1):
            Iteration_buffer[veh_id] = Veh_obj_dict[veh_id].Iteration_Buffer_Out()  # Update Iteration Buffer

        for veh_id in range(1, num_veh + 1):
            flag_lambda[veh_id-1] = Veh_obj_dict[veh_id].Compute_Dual(tau, Pos_buffer, Vel_buffer, Iteration_buffer)

        sum_flag_lambda = sum(flag_lambda)

    for veh_id in range(1, num_veh + 1):
        Iteration_buffer[veh_id] = Veh_obj_dict[veh_id].Iteration_Buffer_Out()  # Update Iteration Buffer
        Veh_obj_dict[veh_id].Pop_u_list()

    for veh_id in range(1, num_veh + 1):
        Veh_obj_dict[veh_id].Compute_Next_State()
        Veh_obj_dict[veh_id].Update_Ego_State()

for veh_id in range(1, num_veh + 1):
    Veh_obj_dict[veh_id].Update_Ego_Traj_Df()

# Output all vehicles' trajectories
Veh_traj_dict = {}
for veh_id in range(num_veh + 1):
    if veh_id == 0:
        Veh_traj_dict[veh_id] = lead_traj
    else:
        Veh_traj_dict[veh_id] = Veh_obj_dict[veh_id].ego_traj

# Plots
## Position
df_position = pd.DataFrame()
for i in range(num_veh+1):
    stri = "PositionPlatoon_"+str(i)
    if i == 0:
        df_position[stri] = Veh_traj_dict[i]['Prec_Pos']
    else:
        df_position[stri] = Veh_traj_dict[i]['Ego_Pos']

df_position = df_position.iloc[:len(lead_traj), :]
df_position.plot()

## Velocity
df_velocity = pd.DataFrame()
for i in range(num_veh+1):
    stri = "VelocityPlatoon_"+str(i)
    if i == 0:
        df_velocity[stri] = Veh_traj_dict[i]['Prec_Vel']
    else:
        df_velocity[stri] = Veh_traj_dict[i]['Ego_Vel']

df_velocity = df_velocity.iloc[:len(lead_traj), :]
df_velocity.plot()

## Acceleration
df_acc = pd.DataFrame()
for i in range(num_veh+1):
    stri = "AccelerationPlatoon_"+str(i)
    if i == 0:
        df_acc[stri] = Veh_traj_dict[i]['Prec_Acc']
    else:
        df_acc[stri] = Veh_traj_dict[i]['Ego_Acc']

df_acc = df_acc.iloc[:len(lead_traj), :]
df_acc.plot()

## SpaceHeadway
df_spacehead = pd.DataFrame()
q = list(df_position.columns)
for i in range(1, df_position.shape[1]):
    stri = "Headspace_"+str(i)
    df_spacehead[stri] = df_position[q[i-1]]-df_position[q[i]]

df_spacehead = df_spacehead.iloc[:len(lead_traj), :]
df_spacehead.plot()

## TimeHeadway
df_timehead = pd.DataFrame()
q = list(df_spacehead.columns)
r = list(df_velocity.columns)
for i in range(1, df_velocity.shape[1]):
    stri = "TimeHead_"+str(i)
    # print(df_velocity[r[i]])
    df_timehead[stri] = df_spacehead[q[i-1]] / df_velocity[r[i]]
    # print(df_timehead)

df_timehead = df_timehead.iloc[:len(lead_traj), :]
df_timehead.plot()












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
tau = 0.017    # control step sampling time

alpha = np.zeros(num_veh)
beta = np.zeros(num_veh)
for veh_id in range(num_veh):
    alpha[veh_id] = 0.1*pow(num_veh, 2)-0.6*(num_veh-veh_id)      # alpha_i = 0.1n^2-0.6(n+1-i), penalty parameter alpha
    beta[veh_id] = 0.3*pow(num_veh, 2)-1.2*(num_veh-veh_id)

D_alpha = np.diag(alpha)           # diagonal matrix of penalty weight alpha
D_beta = np.diag(beta)

E = np.ones((num_veh, num_veh))
S = np.tril(E)                               # Genarate the lower triangular matrix S in the paper
M = np.dot(np.transpose(S), S)               # Calculate transpose(S)*S in the paper
eig_value, eig_vectors = np.linalg.eig(M)    # Calculate the eigenvalues and eigenvectors
P = np.transpose(eig_vectors)                # Orthogonal matrix P in the paper
Q_alpha = np.transpose(P)@D_alpha@P          # Penalty matrix Qz  in the paper
Q_beta = np.transpose(P)@D_beta@P            # Penalty matrix Qz' in the paper

H = np.transpose(np.linalg.inv(S))@(0.25*pow(tau, 4)*Q_alpha+pow(tau, 2)*Q_beta)@np.linalg.inv(S)\
    + pow(tau, 2)*np.eye(num_veh)   # Calculate the matrix of H

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

    Vel_buffer[0] = lead_vel_array[control_step]        # The first value of all buffers is the value from preceding vehicle at time k
    Pos_buffer[0] = lead_pos_array[control_step]
    Iteration_buffer[0] = lead_acc_array[control_step]

    for veh_id in range(1, num_veh+1):
        Vel_buffer[veh_id] = Veh_obj_dict[veh_id].Vel_Buffer_Out()  # Update Velocity Buffer
        Pos_buffer[veh_id] = Veh_obj_dict[veh_id].Pos_Buffer_Out()  # Update Position Buffer

    for veh_id in range(1, num_veh+1):
        Veh_obj_dict[veh_id].Initialize_Controller(tau, control_step, S, Q_alpha, Q_beta, Pos_buffer, Vel_buffer)

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












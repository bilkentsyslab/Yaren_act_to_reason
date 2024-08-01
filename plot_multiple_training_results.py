import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as plb
plb.rcParams['font.size'] = 12
from pathlib import Path

###########################
# Parameters to configure #
###########################
# paths/levels/labes are orderred respectively
# Each index refers to a trained agent's path/level/label (to be used in the legend of each plot)
prefix = os.path.split(os.getcwd())[0]+"/experiments"

paths = [prefix+'/some_title/level1/',
         prefix+'/some_title/level2/',
         prefix+'/some_title/level3/',
         prefix+'/some_title/dynamic_16v/',
         prefix+'/some_title/dynamic_sin/',
         prefix+'/some_title/dynamic/']
levels = ["level1", 
          "level2", 
          "level3",
          "dynamic",
          "dynamic",
          "dynamic"]
labels = ["level1", 
          "level2", 
          "level3",
          "dynamic_16v",
          "dynamic_sin",
          "dynamic_greedy"]

# Directory where the plots and information text will be saved
DIR = prefix+"/some_title/training_results/"

# Information to be written in a file saved in the same directory with the plots
file_info = 'Paper reward'+'\n'+\
    'Sinusoidal (4 to 24) for level-k'+'\n'+\
    '16 vehicles/sinusiodal/greedy for different dynamic agents'+'\n'+\
    'T Progression was at fault before'+'\n'+\
    'T linearly decays to 1 from 50 in 50 states'+'\n'

random_action_subplot = [3,2] # Subplot structure of the greedy vs random action progression plot

Nw_rew_vs_step = 1000 # Averaging window size for "Average Reward vs Step" plot
Nw_rew_vs_ep = 1 # Averaging window size for "Average Reward vs Episode" plot
Nw_rew_per_ep = 100 # Averaging window size for "Average Reward per Episode" plot
Nw_Qloss = 1000 # Averaging window size for "Q-Loss Progression" plot

MAX_STEP = 1200000 # Maximum number of steps taken in the analyzed trainings
FINAL_EP = 10000 # Number of training episodes
###########################

Path("./"+DIR).mkdir(parents=True, exist_ok=True)
for i in range(len(paths)):
    file_info += '' + paths[i] + ': \t' + labels[i] +'\n'
file = open(DIR+"info.txt",'w')
file.write(file_info)
file.close()

all_data_dir = {}
for path_idx, path in enumerate(paths):
    # path_ = path + agent_type +'/training/training_data/'
    # path_ = path +'/training_data/'
    path_ = path +'/training/training_data/'
    agent_type = levels[path_idx]
    
    if os.path.exists(path_+"merged_reward.dat"):
        with open(path_+"merged_reward.dat") as f:
            reward = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float32')
    else:
        with open(path_+"reward.dat") as f:
            reward = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float32')   
            
    if os.path.exists(path_+"merged_train_hist.dat"):
        with open(path_+"merged_train_hist.dat") as f:
            train_hist = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float32')
    else:
        with open(path_+"train_hist.dat") as f:
            train_hist = np.asarray(list(csv.reader(f, delimiter="\t")),dtype='float32')  
            
    if os.path.exists(path_+"merged_ego.csv"):
        ego_df = pd.read_csv(path_+"merged_ego.csv")
        temp_FINAL_EP = ego_df.iloc[-1].Episode.item()
        ep_arr = np.zeros((temp_FINAL_EP,5)) #Argmax-Inf, Argmax, Random, Random-Argmax, Episode
        for episode in range(temp_FINAL_EP):
            ep_df = ego_df.loc[ego_df['Episode']==episode]
            ep_arr[episode,4] = episode
            if episode == 0:
                if agent_type=="dynamic":
                    ep_arr[episode,0] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax-Inf"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,1] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,2] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Random"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,3] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Random-Argmax"].Dynamic_Action_Type.shape[0]
                else:
                    ep_arr[episode,0] = ep_df.loc[ep_df['Action_Type']=="Argmax-Inf"].Action_Type.shape[0]
                    ep_arr[episode,1] = ep_df.loc[ep_df['Action_Type']=="Argmax"].Action_Type.shape[0]
                    ep_arr[episode,2] = ep_df.loc[ep_df['Action_Type']=="Random"].Action_Type.shape[0]
                    ep_arr[episode,3] = ep_df.loc[ep_df['Action_Type']=="Random-Argmax"].Action_Type.shape[0]
            else:
                ep_arr[episode,:4] = ep_arr[episode-1,:4]
                if agent_type=="dynamic":
                    ep_arr[episode,0] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax-Inf"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,1] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,2] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,3] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random-Argmax"].Dynamic_Action_Type.shape[0]
                else:
                    ep_arr[episode,0] += ep_df.loc[ep_df['Action_Type']=="Argmax-Inf"].Action_Type.shape[0]
                    ep_arr[episode,1] += ep_df.loc[ep_df['Action_Type']=="Argmax"].Action_Type.shape[0]
                    ep_arr[episode,2] += ep_df.loc[ep_df['Action_Type']=="Random"].Action_Type.shape[0]
                    ep_arr[episode,3] += ep_df.loc[ep_df['Action_Type']=="Random-Argmax"].Action_Type.shape[0]
    else:
        ego_df = pd.read_csv(path_+"ego.csv")
        temp_FINAL_EP = ego_df.iloc[-1].Episode.item()
        ep_arr = np.zeros((temp_FINAL_EP,5)) #Argmax-Inf, Argmax, Random, Random-Argmax, Episode
        for episode in range(temp_FINAL_EP):
            ep_df = ego_df.loc[ego_df['Episode']==episode]
            ep_arr[episode,4] = episode
            if episode == 0:
                if agent_type=="dynamic":
                    ep_arr[episode,0] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax-Inf"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,1] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,2] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Random"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,3] = ep_df.loc[ep_df['Dynamic_Action_Type']=="Random-Argmax"].Dynamic_Action_Type.shape[0]
                else:
                    ep_arr[episode,0] = ep_df.loc[ep_df['Action_Type']=="Argmax-Inf"].Action_Type.shape[0]
                    ep_arr[episode,1] = ep_df.loc[ep_df['Action_Type']=="Argmax"].Action_Type.shape[0]
                    ep_arr[episode,2] = ep_df.loc[ep_df['Action_Type']=="Random"].Action_Type.shape[0]
                    ep_arr[episode,3] = ep_df.loc[ep_df['Action_Type']=="Random-Argmax"].Action_Type.shape[0]
            else:
                ep_arr[episode,:4] = ep_arr[episode-1,:4]
                if agent_type=="dynamic":
                    ep_arr[episode,0] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax-Inf"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,1] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,2] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random"].Dynamic_Action_Type.shape[0]
                    ep_arr[episode,3] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random-Argmax"].Dynamic_Action_Type.shape[0]
                else:
                    ep_arr[episode,0] += ep_df.loc[ep_df['Action_Type']=="Argmax-Inf"].Action_Type.shape[0]
                    ep_arr[episode,1] += ep_df.loc[ep_df['Action_Type']=="Argmax"].Action_Type.shape[0]
                    ep_arr[episode,2] += ep_df.loc[ep_df['Action_Type']=="Random"].Action_Type.shape[0]
                    ep_arr[episode,3] += ep_df.loc[ep_df['Action_Type']=="Random-Argmax"].Action_Type.shape[0]

    reward_raw = reward.copy()
    reward_df = pd.DataFrame(data=reward_raw,columns=["Episode","Step","Reward"])
    reward_df_eps = reward_df[reward_df.duplicated(subset='Episode',keep='first')==False].Episode.values
    reward2 = np.zeros((len(reward_df_eps),2))
    for episode in reward_df_eps:
        episode = int(episode)
        reward2[episode,0] = episode 
        reward2[episode,1] = reward_df.loc[reward_df["Episode"]==episode].Reward.mean()
    
    reward_ave = np.zeros((reward_df.shape[0],2))
    reward = np.zeros((len(reward_df_eps),2))
    prev_ep = 0
    for episode,step,step_reward in zip(reward_df["Episode"],reward_df["Step"],reward_df["Reward"]):
        episode = int(episode)
        step = int(step)-1
        reward_ave[step,0] = step
        if step == 0:
            reward_ave[step,1] = step_reward
        else:
            reward_ave[step,1] = reward_ave[step-1,1] + (step_reward - reward_ave[step-1,1]) / (step+1.0)
        if episode != prev_ep:
            reward[episode-1,0] = episode-1
            reward[episode-1,1] = reward_ave[step-1,1]
        prev_ep = episode
    reward[-1,0] = prev_ep
    reward[-1,1] = reward_ave[-1,1]
    
    all_data = [reward,reward_ave,reward2,train_hist,ep_arr]

    all_data_dir[path] = all_data
    
del all_data

fig = plt.figure(figsize=(20.0,10.0))
for idx_path,path in enumerate(paths):
    # smoothened = np.convolve(np.asarray(all_data_dir[path][1][:,1]),np.ones(1000)/1000,mode='same')
    if Nw_rew_vs_step > 1:
        smoothened = np.cumsum(np.asarray(all_data_dir[path][1][:,1]), dtype=float)
        smoothened[Nw_rew_vs_step:] = smoothened[Nw_rew_vs_step:] - smoothened[:-Nw_rew_vs_step]
        smoothened[Nw_rew_vs_step-1:] = smoothened[Nw_rew_vs_step- 1:] / Nw_rew_vs_step
        smoothened[:Nw_rew_vs_step-1] = np.divide(smoothened[:Nw_rew_vs_step-1],np.arange(Nw_rew_vs_step-1)+1.0)
    else:
        smoothened = np.asarray(all_data_dir[path][1][:,2], dtype=float)

    plt.plot(np.asarray(all_data_dir[path][1][:,0],dtype='int'),smoothened,
              label=labels[idx_path],alpha=0.7)
plt.title("Training: Average Reward Progression")
plt.ylabel("Average Reward")
plt.xlabel("Step")
plt.grid()
plt.legend()
plt.xlim(0,MAX_STEP)
# plt.ylim(0.01,-0.03)
plt.savefig(DIR+'Training Average Reward vs Step')

fig = plt.figure(figsize=(20.0,10.0))
for idx_path,path in enumerate(paths):
    # smoothened = np.convolve(np.asarray(all_data_dir[path][2][:,1]),np.ones(1)/1,mode='same')
    if Nw_rew_per_ep > 1:
        smoothened = np.cumsum(np.asarray(all_data_dir[path][2][:,1]), dtype=float)
        smoothened[Nw_rew_per_ep:] = smoothened[Nw_rew_per_ep:] - smoothened[:-Nw_rew_per_ep]
        smoothened[Nw_rew_per_ep-1:] = smoothened[Nw_rew_per_ep- 1:] / Nw_rew_per_ep
        smoothened[:Nw_rew_per_ep-1] = np.divide(smoothened[:Nw_rew_per_ep-1],np.arange(Nw_rew_per_ep-1)+1.0)   
    else:
        smoothened = np.asarray(all_data_dir[path][2][:,2], dtype=float)

    plt.plot(np.asarray(all_data_dir[path][2][:,0],dtype='int'),smoothened,
              label=labels[idx_path],alpha=0.7)
plt.title("Training: Average Reward per Episode")
plt.ylabel("Reward per Episode")
plt.xlabel("Episode")
plt.grid()
plt.legend()
plt.xlim(0,FINAL_EP)
plt.savefig(DIR+'Training Average Reward per Episode')

fig = plt.figure(figsize=(20.0,10.0))
for idx_path,path in enumerate(paths):
    # smoothened = np.convolve(np.asarray(all_data_dir[path][3][:,2]), np.ones(1000)/1000,mode='same')
    if Nw_Qloss > 1:
        smoothened = np.cumsum(np.asarray(all_data_dir[path][3][:,2]), dtype=float)
        smoothened[Nw_Qloss:] = smoothened[Nw_Qloss:] - smoothened[:-Nw_Qloss]
        smoothened[Nw_Qloss-1:] = smoothened[Nw_Qloss- 1:] / Nw_Qloss
        smoothened[:Nw_Qloss-1] = np.divide(smoothened[:Nw_Qloss-1],np.arange(Nw_Qloss-1)+1.0)    
    else:
        smoothened = np.asarray(all_data_dir[path][3][:,2], dtype=float)
    plt.plot(np.asarray(all_data_dir[path][3][:,1],dtype='int'), smoothened, 
              label=labels[idx_path],alpha=0.7)
plt.title("Training: Q-Loss Progression")
plt.ylabel("Q-Loss")
plt.xlabel("Step")
plt.grid()
plt.legend()
plt.xlim(0,MAX_STEP)
plt.savefig(DIR+'Training Q-Loss Progression')

fig = plt.figure(figsize=(20.0,10.0))
for idx_path,path in enumerate(paths):
    if Nw_rew_vs_ep > 1:
        smoothened = np.cumsum(np.asarray(all_data_dir[path][0][:,1]), dtype=float)
        smoothened[Nw_rew_vs_ep:] = smoothened[Nw_rew_vs_ep:] - smoothened[:-Nw_rew_vs_ep]
        smoothened[Nw_rew_vs_ep-1:] = smoothened[Nw_rew_vs_ep- 1:] / Nw_rew_vs_ep
        smoothened[:Nw_rew_vs_ep-1] = np.divide(smoothened[:Nw_rew_vs_ep-1],np.arange(Nw_rew_vs_ep-1)+1.0)   
    else:
        smoothened = np.asarray(all_data_dir[path][0][:,1], dtype=float)
    plt.plot(np.asarray(all_data_dir[path][0][:,0],dtype='int'),smoothened,
              label=labels[idx_path],alpha=0.7) 
plt.title("Training: Average Reward vs Episode")
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.grid()
plt.legend()
plt.xlim(0,FINAL_EP)
# plt.ylim(0.01,-0.01)
plt.savefig(DIR+'Training Average Reward vs Episode')

fig, axs = plt.subplots(2, 2, figsize=(20.0,10.0))
for i in range(2):
    for j in range(2):
        for idx_path,path in enumerate(paths):
            axs[i,j].plot(np.asarray(all_data_dir[path][3][:,1],dtype='int'),
                          all_data_dir[path][3][:,3+4*i+2*j], 
                          label=labels[idx_path],
                          alpha=0.7) 
        axs[i,j].set_title("Layer-"+str(i*2+j+1))
        axs[i,j].set(xlabel='Step', ylabel='Average Weight')
        axs[i,j].set_xlim([0,MAX_STEP])
        axs[i,j].grid()
        if i == 0 and j == 0:
            axs[i,j].legend()
fig.savefig(DIR+'Training Average Weight Progression')

fig, axs = plt.subplots(2, 2, figsize=(20.0,10.0))
for i in range(2):
    for j in range(2):
        for idx_path,path in enumerate(paths):
            axs[i,j].plot(np.asarray(all_data_dir[path][3][:,1],dtype='int'),
                          all_data_dir[path][3][:,4+4*i+2*j], 
                          label=labels[idx_path],
                          alpha=0.7) 
        axs[i,j].set_title("Layer-"+str(i*2+j+1))
        axs[i,j].set(xlabel='Step', ylabel='Average Bias')
        axs[i,j].set_xlim([0,MAX_STEP])
        axs[i,j].grid()
        if i == 0 and j == 0:
            axs[i,j].legend()
fig.savefig(DIR+'Training Average Bias Progression')


fig, axs = plt.subplots(random_action_subplot[0], random_action_subplot[1], figsize=(20.0,10.0))
for i in range(random_action_subplot[0]):
    for j in range(random_action_subplot[1]):
        path_no = i*random_action_subplot[1]+j
        if path_no+1 > len(paths):
            break
        path = paths[path_no]
        if random_action_subplot[0] == 1 or random_action_subplot[1] == 1:
            ax = axs[path_no]
        else:
            ax = axs[i,j]
        ax.plot(all_data_dir[path][4][:,4], all_data_dir[path][4][:,0], label="Argmax-Inf")
        ax.plot(all_data_dir[path][4][:,4], all_data_dir[path][4][:,1], label="Argmax")
        ax.plot(all_data_dir[path][4][:,4], all_data_dir[path][4][:,2], label="Random")
        ax.plot(all_data_dir[path][4][:,4], all_data_dir[path][4][:,3], label="Random-Argmax")
        ax.set_title(labels[path_no])
        ax.set_xlim([0,FINAL_EP])
        ax.grid()
        if i == 0 and j == 0:
            ax.legend()
        if i == random_action_subplot[0]//2 and j == 0:
            ax.set(ylabel='Cumulative Number of Actions')
        if i == random_action_subplot[0]-1 and j == random_action_subplot[1]//2:
            ax.set(xlabel='Episode No')
fig.savefig(DIR+'Training Action Type Progression')


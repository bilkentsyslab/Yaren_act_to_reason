import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as plb
plb.rcParams['font.size'] = 12
from pathlib import Path
from operator import add
from Params import Params

###########################
# Parameters to configure #
###########################
# paths/ego_names/ego_types/vs_types/vs_types_fig are orderred respectively
# Each index refers to a simulated agent's 
# --path where the simulation results are saved
# --ego_name to be written in the legend of every plot and comparison.csv
# --ego_types indicates the level and the model of each ego agent (also used to name simulation files)
# --vs_types indicate the level-k environments' type and model where the ego agent is simulated
# --vs_types_fig indicate the names to describe the level-k environments
prefix = os.path.split(os.getcwd())[0]+"/experiments"

paths = [prefix+"/some_title/dynamic_sin/",
         prefix+"/some_title/dynamic/",
         prefix+"/some_title/dynamic/",
         prefix+"/some_title/dynamic_sin/"]

ego_names = ["dynM97_sin",
             "dynM97_gre",
             "dynM98_gre",
             "dyn_rnd"]

ego_types = ["dynamic_m97",
             "dynamic_m97",
             "dynamic_m98",
             "dynamic_random"]

vs_types = [["dynamic_m97","L0","L1_m97","L2_m95","L3_m96"],
            ["dynamic_m97","L0","L1_m97","L2_m95","L3_m96"],
            ["dynamic_m98","L0","L1_m97","L2_m95","L3_m96"],
            ["dynamic_random","L0","L1_m97","L2_m95","L3_m96"]]

vs_types_fig = ["dynamic",
                "L0",
                "L1_m97",
                "L2_m95",
                "L3_m96"]

# Directory where the plots and comparison.csv will be saved
DIR = prefix+"/some_title/simulation_results/dynamic/"

pop_types = [12,16,20,24] # Population groups where the ego agents are trained
num_episodes = 500 # Number of episodes simulated for each population group
dynamic = True # True if the analyzed agents are dynamic agents
###########################

# Dataframe columns for comparison.csv
if dynamic:
    DF_COLUMN = ["Ego_Type", "VS_Type", "Pop_Type", "Velocity_Mean", "Acceleration_Mean", "FC_D_Mean", "Reward_Mean",
                 "Velocity_STD", "Acceleration_STD", "FC_D_STD", "Reward_STD", "L1", "L2", "L3"]
else:
    DF_COLUMN = ["Ego_Type", "VS_Type", "Pop_Type", "Velocity_Mean", "Acceleration_Mean", "FC_D_Mean", "Reward_Mean",
                 "Velocity_STD", "Acceleration_STD", "FC_D_STD", "Reward_STD"]

Path("./"+DIR).mkdir(parents=True, exist_ok=True)

result_df = pd.DataFrame(columns=DF_COLUMN)

for ego_idx, ego_type in enumerate(ego_types):
    vs_ = vs_types[ego_idx]
    path = paths[ego_idx]
    ego_name = ego_names[ego_idx]
    for vs_idx, vs_type in enumerate(vs_):
        sim_df_name = "simulation_"+ ego_type + "_vs_" + vs_type + "_" + str(len(pop_types)) +\
            "x" + str(num_episodes) + "eps_ego.csv"
            
        sim_df = pd.read_csv(path + "simulation/" + sim_df_name)
        
        for pop_idx, pop in enumerate(pop_types):
            eps_df = sim_df.loc[(sim_df['Episode']<(pop_idx+1)*num_episodes)&(sim_df['Episode']>=pop_idx*num_episodes)]
            # eps_df_ave = eps_df.groupby(['Episode']).mean()
            # eps_df_std = eps_df.groupby(['Episode']).std()
            ave_reward = eps_df.Reward.mean()
            ave_vel = eps_df.velocity.mean() * Params.max_speed
            ave_acc = eps_df.Acceleration_X.mean()
            ave_fc_d = eps_df.fc_d.mean() * Params.max_sight_distance
                
            std_reward = eps_df.Reward.std()
            std_vel = eps_df.velocity.std() * Params.max_speed
            std_acc = eps_df.Acceleration_X.std()
            std_fc_d = eps_df.fc_d.std() * Params.max_sight_distance
            
            if dynamic:
                l1_action_num = eps_df.loc[(eps_df['Dynamic_Action']==0)].shape[0]
                l2_action_num = eps_df.loc[(eps_df['Dynamic_Action']==1)].shape[0]
                l3_action_num = eps_df.loc[(eps_df['Dynamic_Action']==2)].shape[0]

                row = [ego_name, vs_type, pop, ave_vel, ave_acc, ave_fc_d,
                        ave_reward, std_vel, std_acc, std_fc_d, std_reward,
                        l1_action_num, l2_action_num, l3_action_num]
                result_df = result_df._append(pd.DataFrame(list([row]),
                                              columns=DF_COLUMN),ignore_index=True)
            else:
                row = [ego_name, vs_type, pop, ave_vel, ave_acc, ave_fc_d, ave_reward,
                       std_vel, std_acc, std_fc_d, std_reward]
                result_df = result_df._append(pd.DataFrame(list([row]),
                                              columns=DF_COLUMN),ignore_index=True)          
            
result_df.to_csv(DIR+"comparison.csv", index=None)

width = 0.2  # the width of the bars

for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        ave_rews = []
        for pop_type in pop_types:
            ave_reward = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Reward_Mean.item()
            ave_rews.append(ave_reward)
        ax.bar(x + ego_idx*width, ave_rews, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Reward')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"ave_reward_"+str(vs_type_fig))
    
for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        ave_vels = []
        for pop_type in pop_types:
            ave_vel = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Velocity_Mean.item()
            ave_vels.append(ave_vel)
        ax.bar(x + ego_idx*width, ave_vels, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Velocity [m/s]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"ave_vel_"+str(vs_type_fig))

for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        ave_accs = []
        for pop_type in pop_types:
            ave_acc = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Acceleration_Mean.item()
            ave_accs.append(ave_acc)
        ax.bar(x + ego_idx*width, ave_accs, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Acceleration [m/s^2]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"ave_acc_"+str(vs_type_fig))
    
for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        ave_fc_ds = []
        for pop_type in pop_types:
            ave_fc_d = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].FC_D_Mean.item()
            ave_fc_ds.append(ave_fc_d)
        ax.bar(x + ego_idx*width, ave_fc_ds, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average FC_D [m]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
    plt.savefig(DIR+"ave_fc_d_"+str(vs_type_fig))

if dynamic:
    for vs_idx, vs_type_fig in enumerate(vs_types_fig):
        levels = ["L1","L2","L3"] 
        fig, ax = plt.subplots()
        x = np.arange(len(ego_names))  # the label locations
        for pop_idx, pop_type in enumerate(pop_types):
            dyn_acts = {"L1":[],"L2":[],"L3":[]}
            for ego_idx, ego_type in enumerate(ego_types):
                vs_type = vs_types[ego_idx][vs_idx]
                ego_name = ego_names[ego_idx]
                row = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                        result_df['VS_Type']==vs_type)&(
                            result_df['Pop_Type']==pop_type)]
                sum_acts = row.L1.item()+row.L2.item()+row.L3.item()*1.0
                dyn_acts['L1'].append(row.L1.item()/sum_acts)
                dyn_acts['L2'].append(row.L2.item()/sum_acts)
                dyn_acts['L3'].append(row.L3.item()/sum_acts)

            if pop_idx == 0:
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L1'], width, color='b',edgecolor='k',label="Level1")
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L2'], width,bottom=dyn_acts['L1'], color='g',edgecolor='k',label="Level2")
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L3'], width,
                        bottom=list( map(add, dyn_acts['L1'], dyn_acts['L2'])), 
                        color='r',edgecolor='k',label="Level3")
            else:
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L1'], width, color='b',edgecolor='k')
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L2'], width,bottom=dyn_acts['L1'], color='g',edgecolor='k')
                ax.bar(x+(pop_idx-1.5)*width, dyn_acts['L3'], width,
                        bottom=list( map(add, dyn_acts['L1'], dyn_acts['L2'])), 
                        color='r',edgecolor='k')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Ratio of Dynamic Strategy')
        ax.set_xlabel('Agent Type')
        # for pop_type in pop_types:
        #     ax.bar_label([pop_type for i in range(len(ego_types))], padding=3)
        ax.set_title("VS: "+vs_type_fig)
        ax.set_xticks(x)
        ax.set_xticklabels(ego_names)
        ax.grid()
        ax.legend()
        plt.show()
        plt.savefig(DIR+"dynamic_action_vs_"+vs_type_fig)
        
for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        std_rews = []
        for pop_type in pop_types:
            std_reward = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Reward_STD.item()
            std_rews.append(std_reward)
        ax.bar(x + ego_idx*width, std_rews, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD Reward')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"std_reward_"+str(vs_type_fig))
    
for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        std_vels = []
        for pop_type in pop_types:
            std_vel = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Velocity_STD.item()
            std_vels.append(std_vel)
        ax.bar(x + ego_idx*width, std_vels, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD Velocity [m/s]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"std_vel_"+str(vs_type_fig))

for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        std_accs = []
        for pop_type in pop_types:
            std_acc = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].Acceleration_STD.item()
            std_accs.append(std_acc)
        ax.bar(x + ego_idx*width, std_accs, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD Acceleration [m/s^2]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    plt.show()
    plt.savefig(DIR+"std_acc_"+str(vs_type_fig))

for vs_idx, vs_type_fig in enumerate(vs_types_fig):
    fig, ax = plt.subplots()
    x = np.arange(len(pop_types))  # the label locations
    for ego_idx, ego_type in enumerate(ego_types):
        vs_type = vs_types[ego_idx][vs_idx]
        ego_name = ego_names[ego_idx]
        std_fc_ds = []
        for pop_type in pop_types:
            std_fc_d = result_df.loc[(result_df['Ego_Type']==ego_name)&(
                result_df['VS_Type']==vs_type)&(
                    result_df['Pop_Type']==pop_type)].FC_D_STD.item()
            std_fc_ds._append(std_fc_d)
        ax.bar(x + ego_idx*width, std_fc_ds, width, label=ego_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD FC_D [m]')
    ax.set_xlabel('Population')
    ax.set_title("VS: "+vs_type_fig)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_types)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()
    plt.savefig(DIR+"std_fc_d_"+str(vs_type_fig))
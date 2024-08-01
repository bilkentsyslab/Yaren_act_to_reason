import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Params import Params
import os

path = os.path.split(os.getcwd())[0]+'/experiments/some_title/level1/training/training_data';
ego_df = pd.read_csv(path+"merged_ego.csv")

actions = {0:"main",1:"acc",2:"de",3:"hacc",4:"hde",5:"mer"}

MAX_T = 50.0
MIN_T = 1.0
FINAL_EP = 10000

dynamic = False # Consider dynamic agent

# Analyze Random/Random-Argmax/Argmax Action Progression
analyze_random_action = True

# Analyze Stopping Cases
analyze_stopping = True

end_after_stopping = False # Agent trained by ending an episode after stopping
before_stopping = False # Check what happened before stopping case

if analyze_stopping:
    if end_after_stopping:
        ego_df_stopping = ego_df.loc[ego_df['State']>=6]
    elif not before_stopping:
        ego_df_stopping = ego_df.loc[ego_df['velocity']<=(-Params.hard_decel_rate*Params.timestep/Params.max_speed)]
    
    ps_stop = []
    ps_merge_stop = []
    
    T = MAX_T
    if not before_stopping:
        for index, row in ego_df_stopping.iterrows():
            record_type = None # 1: Stop, 2: Merge
            if row['Episode']%100 == 0 and row['Episode']>0:                
                T = np.maximum(T - (MAX_T-MIN_T)*(1.0/50), MIN_T)    
            if dynamic:
                for i in range(12,34):  
                    row.iloc[i] = np.exp(row.iloc[i]/T).astype('float64')
                    
                p_ls_sum =  row['q-lev1'] + row['q-lev2'] + row['q-lev3']
                p_l1 = row['q-lev1'] / p_ls_sum
                p_l2 = row['q-lev2'] / p_ls_sum
                p_l3 = row['q-lev3'] / p_ls_sum
                p_l1_sum = row['q-maintain1'] + row['q-accel1'] + \
                    row['q-decel1'] + row['q-hard_accel1'] + \
                        row['q-hard_decel1'] + row['q-merge1']
                p_l2_sum = row['q-maintain2'] + row['q-accel2'] + \
                    row['q-decel2'] + row['q-hard_accel2'] + \
                        row['q-hard_decel2'] + row['q-merge2']
                p_l3_sum = row['q-maintain3'] + row['q-accel3'] + \
                    row['q-decel3'] + row['q-hard_accel3'] + \
                        row['q-hard_decel3'] + row['q-merge3']
            else:
                for i in range(12,18):  
                    row.iloc[i] = np.exp(row.iloc[i]/T).astype('float64')
                p_sum = row['q-maintain'] + row['q-accel'] + \
                    row['q-decel'] + row['q-hard_accel'] + \
                        row['q-hard_decel'] + row['q-merge']
                   
            if end_after_stopping:
                if row['State'] == 6:
                    # print(row['Action'])
                    record_type = 1
                elif row['State'] == 7:
                    record_type = 2
            else:
                vel = row['velocity']*Params.max_speed
                if vel <= -Params.hard_decel_rate*Params.timestep:
                    dist_end_merging = row['dist_end_merging']*Params.merging_region_length
                    fc_d =  row['fc_d']*Params.max_sight_distance
                    
                    # If the headway distance is greater than $far_distance$ while
                    # commuting at a slow speed, record this state
                    if fc_d >= Params.far_distance:
                        record_type = 1
                    # If the ego is on the ramp and close to the end of the merging region
                    # less than $far_distance$, proceed
                    elif row['lane'] == 0 and 0 <= dist_end_merging <= Params.far_distance:
                        fl_d = row['fs_d']*Params.max_sight_distance
                        rl_d = row['rs_d']*Params.max_sight_distance
                        # If there is enough space on the main road, record this state
                        if fl_d >= Params.close_distance and rl_d <= -1.5*Params.far_distance:
                        # if fl_d >= Params.nominal_distance and rl_d <= -1.5*Params.far_distance:
                            record_type = 2
            
            if record_type == 1:
                if dynamic:
                    p_l1_acc = (row['q-accel1'] + row['q-hard_accel1']) / p_l1_sum
                    p_l2_acc = (row['q-accel2'] + row['q-hard_accel2']) / p_l2_sum      
                    p_l3_acc = (row['q-accel3'] + row['q-hard_accel3']) / p_l3_sum
                    p_acc = p_l1 * p_l1_acc + p_l2 * p_l2_acc + p_l3 * p_l3_acc
                else:
                    p_acc = (row['q-accel'] + row['q-hard_accel']) / p_sum
                if end_after_stopping:
                    ps_stop.append([row['Episode'],row['Action'],p_acc])
                else:
                    ps_stop.append([row['Episode'],row['Time_Step'],p_acc])
        
            elif record_type == 2:
                if dynamic:
                    p_l1_merge = row['q-merge1'] / p_l1_sum
                    p_l2_merge = row['q-merge2'] / p_l2_sum      
                    p_l3_merge = row['q-merge3'] / p_l3_sum
                    p_merge = p_l1 * p_l1_merge + p_l2 * p_l2_merge + p_l3 * p_l3_merge
                else:
                    p_merge = row['q-merge'] / p_sum
                if end_after_stopping:
                    ps_merge_stop.append([row['Episode'],row['Action'],p_merge])
                else:
                    ps_merge_stop.append([row['Episode'],row['Time_Step'],p_merge])
            else:
                continue
    
    else:
        for episode in range(FINAL_EP):
            ep_df = ego_df_stopping.loc[ego_df_stopping['Episode']==episode]
            
            for index, row in ep_df.iterrows():
                record_type = None # 1: Stop, 2: Merge
                if row['Episode']%100 == 0 and row['Episode']>0:                
                    T = np.maximum(T - (MAX_T-MIN_T)*(1.0/50), MIN_T)      
                for i in range(12,34):  
                    row.iloc[i] = np.exp(row.iloc[i]/T).astype('float64')    
    
                vel = row['velocity']*Params.max_speed
                if vel <= -Params.hard_decel_rate*Params.timestep:
                    dist_end_merging = row['dist_end_merging']*Params.merging_region_length
                    fc_d =  row['fc_d']*Params.max_sight_distance
                    
                    # If the headway distance is greater than $far_distance$ while
                    # commuting at a slow speed, record this state
                    if fc_d >= Params.far_distance:
                        record_type = 1
                    # If the ego is on the ramp and close to the end of the merging region
                    # less than $far_distance$, proceed
                    elif row['lane'] == 0 and 0 <= dist_end_merging <= Params.far_distance:
                        fl_d = row['fs_d']*Params.max_sight_distance
                        rl_d = row['rs_d']*Params.max_sight_distance
                        # If there is enough space on the main road, record this state
                        if fl_d >= Params.close_distance and rl_d <= -1.5*Params.far_distance:
                        # if fl_d >= Params.nominal_distance and rl_d <= -1.5*Params.far_distance:
                            record_type = 2
                
                if record_type == 1 or record_type == 2:
                    prev_row = ep_df.loc[ep_df['Time_Step']==(row['Time_Step']-1)].iloc[[0]]
                    p_ls_sum =  prev_row['q-lev1'] + prev_row['q-lev2'] + prev_row['q-lev3']
                    p_l1 = prev_row['q-lev1'] / p_ls_sum
                    p_l2 = prev_row['q-lev2'] / p_ls_sum
                    p_l3 = prev_row['q-lev3'] / p_ls_sum
                    p_l1_sum = prev_row['q-maintain1'] + prev_row['q-accel1'] + \
                        prev_row['q-decel1'] + prev_row['q-hard_accel1'] + \
                            prev_row['q-hard_decel1'] + prev_row['q-merge1']
                    p_l2_sum = prev_row['q-maintain2'] + prev_row['q-accel2'] + \
                        prev_row['q-decel2'] + prev_row['q-hard_accel2'] + \
                            prev_row['q-hard_decel2'] + prev_row['q-merge2']
                    p_l3_sum = prev_row['q-maintain3'] + prev_row['q-accel3'] + \
                        prev_row['q-decel3'] + prev_row['q-hard_accel3'] + \
                            prev_row['q-hard_decel3'] + prev_row['q-merge3']
                    
                    if record_type == 1:
                        p_l1_acc = (prev_row['q-accel1'] + prev_row['q-hard_accel1']) / p_l1_sum
                        p_l2_acc = (prev_row['q-accel2'] + prev_row['q-hard_accel2']) / p_l2_sum      
                        p_l3_acc = (prev_row['q-accel3'] + prev_row['q-hard_accel3']) / p_l3_sum
                        p_acc = p_l1 * p_l1_acc + p_l2 * p_l2_acc + p_l3 * p_l3_acc
                        ps_stop.append([prev_row['Episode'].item(),prev_row['Action'].item(),p_acc.item()])
                    elif record_type == 2:
                        p_l1_merge = prev_row['q-merge1'] / p_l1_sum
                        p_l2_merge = prev_row['q-merge2'] / p_l2_sum      
                        p_l3_merge = prev_row['q-merge3'] / p_l3_sum
                        p_merge = p_l1 * p_l1_merge + p_l2 * p_l2_merge + p_l3 * p_l3_merge
                        ps_merge_stop.append([prev_row['Episode'].item(),prev_row['Action'].item(),p_merge.item()])
                        
                    break
                else:
                    continue
    
    ps_stop = np.array(ps_stop)    
    ps_merge_stop = np.array(ps_merge_stop)    
    
    if ps_stop.shape[0] > 1:
        fig = plt.figure(figsize=(20.0,10.0))
        plt.plot(ps_stop[:,0], ps_stop[:,2],'x--')
        for i, txt in enumerate(ps_stop[:,1]):
            if end_after_stopping or before_stopping:
                plt.annotate(str(actions[int(txt)]), (ps_stop[i,0], ps_stop[i,2]))
            else:
                plt.annotate(str(int(txt)), (ps_stop[i,0], ps_stop[i,2]))
        plt.title("Accel/Hard-Accel Probability Progression in Stopping Case")
        plt.ylabel("Probability")
        plt.xlabel("Instance")
        plt.ylim([0,1])
        plt.savefig(path+'Hard Accel Probability Progression in Stopping Case')
    
    if ps_merge_stop.shape[0] > 1:
        fig = plt.figure(figsize=(20.0,10.0))
        plt.plot(ps_merge_stop[:,0], ps_merge_stop[:,2],'x--')
        for i, txt in enumerate(ps_merge_stop[:,1]):
            if end_after_stopping or before_stopping:
                plt.annotate(str(actions[int(txt)]), (ps_merge_stop[i,0], ps_merge_stop[i,2]))
            else:
                plt.annotate(str(int(txt)), (ps_merge_stop[i,0], ps_merge_stop[i,2]))
        plt.title("Merge Probability Progression in Ramp Stopping Case")
        plt.ylabel("Probability")
        plt.xlabel("Instance")
        plt.ylim([0,1])
        plt.savefig(path+'Merge Probability Progression in Ramp Stopping Case')

if analyze_random_action:
    ep_arr = np.zeros((FINAL_EP,5)) #Argmax-Inf, Argmax, Random, Random-Argmax, Episode
    for episode in range(FINAL_EP):
        ep_df = ego_df.loc[ego_df['Episode']==episode]
        ep_arr[episode,4] = episode
        if episode == 0:
            if dynamic:
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
            if dynamic:
                ep_arr[episode,0] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax-Inf"].Dynamic_Action_Type.shape[0]
                ep_arr[episode,1] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Argmax"].Dynamic_Action_Type.shape[0]
                ep_arr[episode,2] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random"].Dynamic_Action_Type.shape[0]
                ep_arr[episode,3] += ep_df.loc[ep_df['Dynamic_Action_Type']=="Random-Argmax"].Dynamic_Action_Type.shape[0]
            else:
                ep_arr[episode,0] += ep_df.loc[ep_df['Action_Type']=="Argmax-Inf"].Action_Type.shape[0]
                ep_arr[episode,1] += ep_df.loc[ep_df['Action_Type']=="Argmax"].Action_Type.shape[0]
                ep_arr[episode,2] += ep_df.loc[ep_df['Action_Type']=="Random"].Action_Type.shape[0]
                ep_arr[episode,3] += ep_df.loc[ep_df['Action_Type']=="Random-Argmax"].Action_Type.shape[0]
                
    
    fig = plt.figure(figsize=(20.0,10.0))
    plt.plot(ep_arr[:,4], ep_arr[:,0], label="Argmax-Inf")
    plt.plot(ep_arr[:,4], ep_arr[:,1], label="Argmax")
    plt.plot(ep_arr[:,4], ep_arr[:,2], label="Random")
    plt.plot(ep_arr[:,4], ep_arr[:,3], label="Random-Argmax")
    plt.annotate(str(ep_arr[-1,0]), (FINAL_EP-1, ep_arr[-1,0],))
    plt.annotate(str(ep_arr[-1,1]), (FINAL_EP-1, ep_arr[-1,1],))
    plt.annotate(str(ep_arr[-1,2]), (FINAL_EP-1, ep_arr[-1,2],))
    plt.annotate(str(ep_arr[-1,3]), (FINAL_EP-1, ep_arr[-1,3],))
    plt.title("Action Type Progression")
    plt.ylabel("Cumulative Number of Actions")
    plt.xlabel("Episode No")
    plt.legend()
    plt.savefig(path+'Action Type Progression')   
    
    
# long_eps = ego_df.loc[ego_df['Time_Step']>=500]
# eps = long_eps[long_eps.duplicated(subset='Episode',keep='first')==False].Episode.values
# print(eps)

# for ep_no in eps:
#     ep_df = ego_df.loc[ego_df['Episode']==ep_no]
#     print("ep_no: "+str(ep_no))
#     print(ep_df.shape[0])
#     # stop_df = ep_df.loc[(ep_df['velocity']*Params.max_speed)<=(-Params.hard_decel_rate*Params.timestep)]
#     # print(stop_df.shape[0])
#     fig = plt.figure(figsize=(20.0,10.0))
#     ep_df['velocity'] *= Params.max_speed
#     ep_df['fc_d'] *= Params.max_sight_distance
#     ep_df['fs_d'] *= Params.max_sight_distance
#     ep_df['rs_d'] *= Params.max_sight_distance
#     ep_df['fc_v'] *= Params.max_speed
#     ep_df['fs_v'] *= Params.max_speed
#     ep_df['rs_v'] *= Params.max_speed
#     ep_df['dist_end_merging'] *= Params.merging_region_length
#     ep_df['lane'] *= 50
#     plt.title("Episode: "+str(ep_no))
#     plt.plot('Time_Step','velocity', data=ep_df)
#     plt.plot('Time_Step','fc_d', data=ep_df)
#     plt.plot('Time_Step','fs_d', data=ep_df)
#     plt.plot('Time_Step','rs_d', data=ep_df)
#     plt.plot('Time_Step','fc_v', data=ep_df)
#     plt.plot('Time_Step','fs_v', data=ep_df)
#     plt.plot('Time_Step','rs_v', data=ep_df)
#     plt.plot('Time_Step','dist_end_merging', data=ep_df)
#     plt.plot('Time_Step','lane', data=ep_df)
#     plt.legend()
#     plt.grid()
#     plt.show()

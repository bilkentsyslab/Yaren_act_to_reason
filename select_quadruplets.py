#Select 4-vehicle groups where the ego IDs are provided beforehand
#Filter the complete dataset into separate episode dataframes where
# each corresponds to the recorded 
# © 2021 Cevahir Köprülü All Rights Reserved
import pandas as pd 
from Params import Params
from pathlib import Path
import os

DATA_PATH = os.path.split(os.getcwd())[0]+'data/'
VEHICLE_PATH = DATA_PATH + 'vehicles/'


DATA = pd.read_csv(DATA_PATH+"RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv")
vs_always6_final_filtered = pd.read_csv(VEHICLE_PATH+"vs_always6_final_filtered.csv")
vs_startsAt7_endsAt6_final_filtered = pd.read_csv(VEHICLE_PATH+"vs_startsAt7_endsAt6_final_filtered.csv")

'''
Quadruplets for vehicles starting on the main lane
--------------------------------------------------

1-Find followers or leaders that are merged from the ramp
2-Find the followers and leaders of the car that merged
3-Find the leaders of the ego
'''

DIR = DATA_PATH+"NGSIM_I80_quadruplets"
Path("./"+DIR).mkdir(parents=True, exist_ok=True)

print("vs_always6_final_filtered")
DIR_VS_6 = "vs_always6"

for EGO_ID in vs_always6_final_filtered.Vehicle_ID:

    print("\nEGO_ID: "+str(EGO_ID))
    EGO_DF = DATA[DATA['Vehicle_ID']==EGO_ID].copy()
    EGO_DF = EGO_DF.loc[EGO_DF['Local_Y']>=((Params.start_merging_point-Params.offset)/0.3048)]
    # ego_df = ego_df.loc[ego_df['Local_Y']<=(Params.end_for_car0/0.3048)]
    first_frame = EGO_DF.iloc[0].Frame_ID
    last_frame = EGO_DF.iloc[-1].Frame_ID
    
    EPISODE_DF = DATA.loc[((DATA['Frame_ID']>=first_frame) & (DATA['Frame_ID']<=last_frame) & ((DATA['Lane_ID'] == 6) | (DATA['Lane_ID'] == 7)))].copy();
    EPISODE_DF = EPISODE_DF.reset_index(drop=True)
    EPISODE_DF.sort_values(by=['Frame_ID'],inplace=True)
    
    ego_leaders = list(EGO_DF[EGO_DF.duplicated(subset='Leader_ID',keep='first')==False].Leader_ID.values)
    ego_followers = EGO_DF[EGO_DF.duplicated(subset='Follower_ID',keep='first')==False].Follower_ID;

    
    Path("./"+DIR+"/with_followers/"+DIR_VS_6+"/"+str(EGO_ID)).mkdir(parents=True, exist_ok=True)
    Path("./"+DIR+"/no_followers/"+DIR_VS_6+"/"+str(EGO_ID)).mkdir(parents=True, exist_ok=True)

    # Leaders #
    if EGO_ID in [2799]:
        print("Leader part is ignored")
    else:
        print("Leader part")
        for leader_id in ego_leaders:
            leader_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == leader_id]
            if (leader_df.Lane_ID == 7).any():
                print(leader_id)
     
                temp_cars = [EGO_ID,leader_id]+ego_leaders
                # Select its leaders and followers
                leader_df_onramp = leader_df.loc[leader_df['Lane_ID'] == 7]
                
                leader_leader_ids = list(leader_df_onramp[leader_df_onramp.duplicated(subset='Leader_ID',keep='first')==False].Leader_ID.values)
                temp_cars += leader_leader_ids
                
                leader_follower_ids = list(leader_df_onramp[leader_df_onramp.duplicated(subset='Follower_ID',keep='first')==False].Follower_ID.values)
                temp_cars += leader_follower_ids
    
                leader_df_onramp_last_frame_id = leader_df_onramp['Frame_ID'].iloc[-1]
                del leader_df_onramp, leader_df
                
                temp_cars = (list(set(temp_cars)))
                minus_1_index = temp_cars.index(-1)
                del temp_cars[minus_1_index]
                
                ep_df = EPISODE_DF[EPISODE_DF['Vehicle_ID'].isin(temp_cars)].copy()
                ep_df.to_csv(DIR+"/with_followers/"+DIR_VS_6+"/"+str(EGO_ID)+"/ep_leader_"+str(leader_id)+".csv", index = None, header=True)
                
                ##################################################
                ### REMOVE (MERGED) FOLLOWERS ON THE MAIN ROAD ###
                ##################################################
                new_df = pd.DataFrame(columns=ep_df.columns)
                ep_frames = ep_df[ep_df.duplicated(subset='Frame_ID',keep='first')==False].Frame_ID
                for frame_id in ep_frames:
                    frame_df = ep_df.loc[ep_df['Frame_ID']==frame_id]
                    ego_frame = frame_df.loc[frame_df['Vehicle_ID']==EGO_ID]
                    frame_df = frame_df.loc[~((frame_df['Local_Y']<ego_frame.Local_Y.item())&(frame_df['Lane_ID']==ego_frame.Lane_ID.item()))].copy()
                    new_df = new_df._append(frame_df)
                ep_df = new_df
                ##################################################
                ### REMOVE (MERGED) FOLLOWERS ON THE MAIN ROAD ###
                ##################################################
                
                ep_df.to_csv(DIR+"/no_followers/"+DIR_VS_6+"/"+str(EGO_ID)+"/ep_leader_"+str(leader_id)+".csv", index = None, header=True)
        
    # Followers #
    # Ignore some egos in order prevent from repetition of scenarios
    if EGO_ID in [146,1066,2344,3006]:
        print("Follower part is ignored")
    else:
        print("Follower part")
        for follower_id in ego_followers.values:
    
            follower_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID'] == follower_id]
            if (follower_df.Lane_ID == 7).any():
                print(follower_id)
                
                temp_cars = [EGO_ID,follower_id]+ego_leaders
                # Select its leaders and followers
                follower_df_onramp = follower_df.loc[follower_df['Lane_ID'] == 7]
                
                follower_leader_ids = list(follower_df_onramp[follower_df_onramp.duplicated(subset='Leader_ID',keep='first')==False].Leader_ID.values)
                temp_cars += follower_leader_ids
                
                # follower_follower_ids = list(follower_df_onramp[follower_df_onramp.duplicated(subset='Follower_ID',keep='first')==False].Follower_ID.values)
                # temp_cars += follower_follower_ids
                
                follower_df_onramp_last_frame_id = follower_df_onramp['Frame_ID'].iloc[-1]
                del follower_df_onramp, follower_df
                        
                temp_cars = (list(set(temp_cars)))
                minus_1_index = temp_cars.index(-1)
                del temp_cars[minus_1_index]
                
                ep_df = EPISODE_DF[EPISODE_DF['Vehicle_ID'].isin(temp_cars)].copy()
                ep_df.to_csv(DIR+"/with_followers/"+DIR_VS_6+"/"+str(EGO_ID)+"/ep_follower_"+str(follower_id)+".csv", index = None, header=True)
    
                ##################################################
                ### REMOVE (MERGED) FOLLOWERS ON THE MAIN ROAD ###
                ##################################################
                new_df = pd.DataFrame(columns=ep_df.columns)
                ep_frames = ep_df[ep_df.duplicated(subset='Frame_ID',keep='first')==False].Frame_ID
                for frame_id in ep_frames:
                    frame_df = ep_df.loc[ep_df['Frame_ID']==frame_id]
                    ego_frame = frame_df.loc[frame_df['Vehicle_ID']==EGO_ID]
                    frame_df = frame_df.loc[~((frame_df['Local_Y']<ego_frame.Local_Y.item())&(frame_df['Lane_ID']==ego_frame.Lane_ID.item()))].copy()
                    new_df = new_df._append(frame_df)
                ep_df = new_df
                ##################################################
                ### REMOVE (MERGED) FOLLOWERS ON THE MAIN ROAD ###
                ##################################################
                
                ep_df.to_csv(DIR+"/no_followers/"+DIR_VS_6+"/"+str(EGO_ID)+"/ep_follower_"+str(follower_id)+".csv", index = None, header=True)

'''
Quadruplets for vehicles starting on the ramp
--------------------------------------------------

1-Find followers and leaders from the main lane
2-Find leaders on the ramp
'''
print("vs_startsAt7_endsAt6_final_filtered")
DIR_VS_7 = "vs_startsat7"
for EGO_ID in vs_startsAt7_endsAt6_final_filtered.Vehicle_ID:
    print("\nEGO_ID: "+str(EGO_ID))
    EGO_DF = DATA[DATA['Vehicle_ID']==EGO_ID].copy()
    EGO_DF = EGO_DF.loc[EGO_DF['Local_Y']>=(Params.start_merging_point-Params.offset)]
    # ego_df = ego_df.loc[ego_df['Local_Y']<=(Params.end_for_car0/0.3048)]
    first_frame = EGO_DF.iloc[0].Frame_ID
    last_frame = EGO_DF.iloc[-1].Frame_ID
    
    EPISODE_DF = DATA.loc[((DATA['Frame_ID']>=first_frame) & (DATA['Frame_ID']<=last_frame) & ((DATA['Lane_ID'] == 6) | (DATA['Lane_ID'] == 7)))].copy();
    EPISODE_DF = EPISODE_DF.reset_index(drop=True)
    EPISODE_DF.sort_values(by=['Frame_ID'],inplace=True)
    
    ego_leaders = list(EGO_DF[EGO_DF.duplicated(subset='Leader_ID',keep='first')==False].Leader_ID.values)

    Path("./"+DIR+"/with_followers/"+DIR_VS_7+"/"+str(EGO_ID)).mkdir(parents=True, exist_ok=True)
    Path("./"+DIR+"/no_followers/"+DIR_VS_7+"/"+str(EGO_ID)).mkdir(parents=True, exist_ok=True)

    EGO_DF_main = EGO_DF.loc[EGO_DF['Lane_ID']==6]
    ego_followers_main = list(EGO_DF_main[EGO_DF_main.duplicated(subset='Follower_ID',keep='first')==False].Follower_ID.values)
    ego_followers_main_leaders = []
    for ego_follower_main in ego_followers_main:
        ego_follower_main_df = EPISODE_DF.loc[EPISODE_DF['Vehicle_ID']==ego_follower_main]
        ego_follower_main_leaders = ego_follower_main_df[ego_follower_main_df.duplicated(subset='Leader_ID',keep='first')==False].Leader_ID
        ego_followers_main_leaders += list(ego_follower_main_leaders.values)
        
    ego_followers_main_leaders = (list(set(ego_followers_main_leaders)))
    del EGO_DF_main

    temp_cars = [EGO_ID]+ego_leaders+ego_followers_main+ego_followers_main_leaders
    temp_cars = (list(set(temp_cars)))

    ep_df = EPISODE_DF[EPISODE_DF['Vehicle_ID'].isin(temp_cars)].copy()
    ep_df.to_csv(DIR+"/with_followers/"+DIR_VS_7+"/"+str(EGO_ID)+"/ep.csv", index = None, header=True)   

    ##########################################################
    ### REMOVE FOLLOWERS ON THE MAIN ROAD AFTER EGO MERGES ###
    ##########################################################
    new_df = pd.DataFrame(columns=ep_df.columns)
    ep_frames = ep_df[ep_df.duplicated(subset='Frame_ID',keep='first')==False].Frame_ID
    for frame_id in ep_frames:
        frame_df = ep_df.loc[ep_df['Frame_ID']==frame_id]
        if frame_df.loc[frame_df['Vehicle_ID']==EGO_ID].Lane_ID.item() == 6:
            ego_frame = frame_df.loc[frame_df['Vehicle_ID']==EGO_ID]
            frame_df = frame_df.loc[~((frame_df['Local_Y']<ego_frame.Local_Y.item())&(frame_df['Lane_ID']==ego_frame.Lane_ID.item()))].copy()
        new_df = new_df._append(frame_df)
    ep_df = new_df
    ##########################################################
    ### REMOVE FOLLOWERS ON THE MAIN ROAD AFTER EGO MERGES ###
    ##########################################################
    
    ep_df.to_csv(DIR+"/no_followers/"+DIR_VS_7+"/"+str(EGO_ID)+"/ep.csv", index = None, header=True)   

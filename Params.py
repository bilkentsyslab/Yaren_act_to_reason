#This implemets a container for parameters.
# © 2020 Cevahir Köprülü All Rights Reserved

class Params:
    timestep = 0.5 # Decision Interval, s 
    
    dynamic_history = 10 #3 #1 # Number of frames given as input to the dynamic agent
    dynamic_history_action = True # Actions of previous frames given as input to the dynamic agent
    scale_action = True # Scale action input of the dynamic history
    
    # Inside the region of interest (position <= 865 ft) in NGSIM I-80
    nominal_speed = 6 # Mean Velocity 5.95 m/s =21.42 kmh || STD = 3.307 m/s
    min_speed = 0 # Minimum Speed
    max_speed = 105/3.6 # Maximum Speed, ~65 mph (I-80 Speed Limit)
    speed_randomness = 1#2

    accel_rate = 2 # Acceleration Rate || Acceleration Distribution: Mean = -0.37 ft/s^2, STD = 3.21 ft/s^2 
    decel_rate = -2 # Deceleration Rate
    hard_accel_rate = 3 # Hard Acceleration Rate (Maximum Acceleration in the distribution: 10 ft/s^2, also ordinary cars can accelerate between 3-4 m/s^2)
    hard_decel_rate = -4.5 # Hard Deceleration Rate (Minimum Deceleration in the distribution: -15 ft/s^2, also ordinary cars can decelerate upto -4.5m/s^2)
    
    carlength = 5
    carwidth = 2
    lanewidth = 3.7 # ~12 feet

    # For pdf analysis, headway limit is set as 70m
    # Mean = 11.9m
    # STD = 9.6m
    close_distance = 2.3
    nominal_distance = 11.9
    far_distance = 21.5
    min_initial_separation = carlength*2
    max_sight_distance = 70
    
    num_observations = 9 # FS/FC/RS x 2 (Relative Distance + Relative Velocity) + Velocity + Current Lane + Distance to end_merging_point
    num_actions = 6 # maintain/accel/decel/hard-accel/hard-decel/merge
    num_dynamic_actions = 3
    
    NGSIM_setup = False
    offset = 0 if NGSIM_setup else 50 # An offset to enlargen the environment
    start_environment = 50 + offset # 165ft
    start_onramp = 75 + offset # 250 ft - From 250th to 720th, it covers around 98.6% of frames recorded on the ramp
    start_merging_point = 120 + offset # 385ft
    end_merging_point = 220 + offset # 720 ft - From 250th to 720th, it covers around 98.6% of frames recorded on the ramp
    merging_region_length = end_merging_point - start_merging_point
    add_car_point = end_merging_point + offset #265 + offset
    end_for_car0 = end_merging_point if NGSIM_setup else (265 + offset)
    init_size = 265 + offset
    
    actions = {0:["maintain",0,0],
               1:["accelerate", accel_rate, 0],
               2:["decelerate", decel_rate, 0],
               3:["hard_accelerate", hard_accel_rate, 0],
               4:["hard_decelerate", hard_decel_rate, 0],
               5:["merge", 0, lanewidth/timestep]}
    
    dynamic_actions = {1:"level-1",
                       2:"level-2",
                       3:"level-3"}
    
    car0_states = {-1:"Moving",
                   0:"Ego Didn't Merge", 
                   1:"A Car Merged into Ego",
                   2:"Ego Merged into a Car",
                   3:"Ego Crashed",
                   4:"Completed Safely",
                   5:"Rear-Ended",
                   6:"Unnecessary Stopping",
                   7:"Ego Should Have Merged"}

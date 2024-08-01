# Act to Reason - Code

This repository includes the Python project created to implement the ***Act to Reason: A Dynamic Game Theoretical Model for Highway Merging Applications*** paper, presented in CCTA 2021, by Cevahir Koprulu and Yildiray Yildiz.


## Script/Class Descriptions

The following describes the purpose of each script/class in short.

### train_and_simulate.py:
Script to train and/or simulate a (dynamic) level-k agent. Configurations can be made here, training/simulation files are to be saved into a directory called experiments.
### visualize_in_GUI.py:
Script to visualize recorded episodes in a custom-made GUI.
### src/Params.py:
Container for parameters used to create the environment for training and simulation of level-k and dynamic agents in the highway merging setting.
### src/Message.py:
Container for message, namely an observation of an agent at a time step.
### src/State.py:
State class object is used to generate a highway merging environment instance. It initializes the environment, moves vehicles with their corresponding actions, keeps track of the state of the environment, contains the reward function and the level-0 policy. 
### src/DQNAgent.py:
Implementation of the Deep Q-Network for a level-k agent.
### src/DynamicDQNAgent.py:
Implementation of the Deep Q-Network for a dynamic level-k agent.
### src/Training.py:
Implementation of the training of level-k/dynamic agents in highway merging setting.
### src/Simulation.py:
Implementation of the simulation of level-k/dynamic agents in highway merging setting.
### src/SimulationAnalyzer.py:
Implementation of the simulation analyzer which evaluates the result of a simulation.
### src/HighwayMergingGUI.py:
Graphical user interface to visualize recorded episodes.
### utils/analyze_ego.py:
Utility script to analyze recorded dataframes of a level-k agent or a dynamic level-k agent in the training. Analysis covers the progression of the probability of taking an action greedily, which is critical to see how long an agent explores the environment, and taking an action to exit a stopping case, which is important to observe whether an agent learns to get itself out of stopping cases throughout the training.
### utils/merge_and_plot.py:
Utility script to merge retraining files, if there are any, and plot the results of a training.
### utils/plot_multiple_training_results.py:
Utility script to plot multiple training results for comparison
-Should be run after merging all retraining files if there are any
### utils/plot_multiple_simulation_egos.py:
Utility script to plot multiple simulation results for ego agents for comparison.
### analysis/select_I80_ego_vehicles.py:
Analysis script to select the initial candidates for ego vehicles from the NGSIM I-80 dataset. This script is the first one to run in order to create necessary dataframes that include information about the selected vehicles which will be replaced by dynamic level-k agents for comparison.
### analysis/filter_and_plot_lane6and7_vehicles.py:
Analysis script to filter more vehicles with respect to their merging position. There are very rare cases where a vehicle merges late due to the structure of the on-ramp in Emeryville, so in order to determine a shorter merging region, these cases can be ignored. In addition, this script can be used to plot how vehicles are distributed in the highway with respect to their positioning in order to determine the limits of the region of interest in the dataset.
### analysis/select_quadruplets.py:
Analysis script to select quadruplets, ego vehicles plus 3 surrounding vehicles, in the dataset. There are two main directories: vs_always6, which refers to the egos that commute on the (rightmost lane of) main road at all times, and vs_startat7, which refers to the egos that start on the ramp and then merge. Each group is separated into 2 further categories: with_followers, where the vehicles behind the ego are kept, and no_followers, where they are ignored in order to prevent collisions occur from the rear of the ego.
### analysis/simulate_in_NGSIM_data.py:
Analysis script to simulate dynamic level-k agents placing them in selected quadruplets as ego vehicles. Models to be simulated can be provided as a list, with the preferred number of runs considering the stochasticity of the models. Simulations can be carried out as complete episodes or as sections (trajectories of a certain number of timesteps) of the episodes where the total duration of a section can be specified as a constant.
### analysis/simGUI_I80_reconstructed.py:
Analysis script to demonstrate the original quadruplet recording and simulations where a dynamic level-k agent is placed as the ego.
### analysis/analyze_NGSIM_sim_results.py:
Analysis script to analyze the simulated episodes where a dynamic level-k agent replaces an ego in selected quadruplates. These simulations are compared to original recordings in terms of position/velocity/acceleration.
### analysis/analyze_NGSIM_sim_results_traj.py:
Analysis script to analyze the simulated section/trajectories where a dynamic level-k agent replaces an ego in selected quadruplates. These simulations are compared to original recordings in terms of position/velocity/acceleration.

## Instructions for Training

Instructions to train a level-k/dynamic agent:
Go to Training.py
    
   1) Set ***path***
   2) Set ***train*** to True
   3) Set ***ego_level*** according to the level of the agent to be trained
   4) Set ***models*** that should be loaded according to the level of the surrounding vehicles
   5) Set ***random_dynamic_strategy*** and ***dynamic_driving_boltzmann_sampling*** according to your choosing if the agent is dynamic
   6) Set ***rnd_levk_env_eps*** and ***dynamic_vs*** according to your choosing if the agent is dynamic
   7) Set the parameters under "TRAINING PART" according to your choosing (Currently set according to the CCTA 2021 paper)
   8) At the end, run merge_and_plot.py with the appropriate parameters to plot training results

If a model is to be retrained.
   1) Set ***retrain*** to True
   2) Set ***first_state_reset*** to the first state where the new training should start
   3) At the end, run *merge_and_plot.py* with the appropriate parameters to merge retraining files and plot training results
#meaning retrain.. = [9500, 98000] if retrain at 95 and 98th sections of hundred.
Note that retraining can be initiated when an agent gets stuck in an episode, the reward begins to drop unreasonably due to excessive amount of collision or long waiting periods.

## Instructions for Simulation

Instructions to simulate a level-k/dynamic agent:
Following instructions are for running simulation in Training.py
    
   1) Set ***path***
   2) Set ***simulate*** to True
   3) Set ***ego_level*** according to the level of the agent to be simulated
   4) Set ***models*** that should be loaded according to the level of the surrounding vehicles
   5) Set ***random_dynamic_strategy*** and ***dynamic_driving_boltzmann_sampling*** according to your choosing if the agent is dynamic
   6) Set ***sim_save_ego_df***, ***sim_ego_models*** and ***sim_vs*** according to your choosing
   7) Set the parameters under "SIMULATION PART" according to your choosing (Currently set according to the CCTA 2021 paper)

Following instructions are for running simulation in Training.py
   1) Set ***path***
   2) Set ***ego_level*** according to the level of the agent to be simulated
   3) Set ***models*** that should be loaded according to the level of the surrounding vehicles
   4) Set ***ego_models*** and ***vs*** to be simulated
   5) Set ***num_episodes*** and ***car_population***
   6) Set the other parameters according to your choosing (Currently set according to the CCTA 2021 paper)

## Tips for selecting a model to train agents of higher levels or a dynamic agent

Selecting a model is a critical part, as it is quite hard to find a level-k model that performs optimally in level-(k-1) environment and also behave relatively well in level-k environment. The latter part is crucial in order to provide a safe environment to train a level-(k+1) agent. In general, one of the last 5 models are selected for further work. 
There are 2 aspects to consider:
    
   1) Average reward and collision rate progression throughout the training
   2) Collision rates in level-(k-1) and level-k environments
    
After the training, run merge_and_plot.py in utils to obtain plots which demonsrate agent's progression throughout the training. Check the average reward progression and the collision rates in the last 5 states, i.e. 500 episodes if a state is of 100 episodes. If there is a sudden drop of reward or increase of collision rates, then retrain from a certain state to be sure that the agent doesn't go through episodes which damage its behavior negatively. If not, then proceed to the next stage.        

As explained before, it's important for a level-k agent to behave well in both of these environments. Let's consider the following cases for a level-1 agent. Suppose that the collision rates for the 98th model are 0.5% and 20% in level-0 and level-1 environments, respectively, whereas they are 1% and 15% for the 99th model. Then, considering the fact that behavior in level-1 environment is critical to train level-2 agents, the 99th model can be determined as the model to use for further work.


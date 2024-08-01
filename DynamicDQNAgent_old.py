#This file implements DQN for Dynamic Agent
# © 2020 Cevahir Köprülü All Rights Reserved

from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam
import tensorflow as tf
from DQNAgent import DQNAgent
import pickle
import random
import numpy as np
import gc
# import tensorflow as tf
# tf.set_random_seed(0)
# random.seed(0)
# np.random.seed(0)

class DynamicDQNAgent:
    def __init__(self, dynamic_state_size, dynamic_action_size, state_size, action_size, levelk_config):
        self.dynamic_state_size = dynamic_state_size
        self.dynamic_action_size = dynamic_action_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000) #experience set, i.e. memory
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.002
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.T_SCALE = 1.0
        self.MIN_T = self.T_SCALE * 1.0
        self.MAX_T = self.T_SCALE * np.float32(50)
        self.T = self.T_SCALE * np.float32(50) #Initial boltzmann temperature
        self.boltzmann_sampling = True # Boltzmann Sampling
        self.dynamic_driving_boltzmann_sampling = levelk_config["boltzmann_sampling"]
        self.random_dynamic_strategy = False # Dynamic agent randomly selects a level-k strategy
        self.agentlevk = {}
        self.agentlevk[0] = self._load_levelk(levelk_config["paths"][1],self.dynamic_driving_boltzmann_sampling)
        self.agentlevk[1] = self._load_levelk(levelk_config["paths"][2],self.dynamic_driving_boltzmann_sampling)
        self.agentlevk[2] = self._load_levelk(levelk_config["paths"][3],self.dynamic_driving_boltzmann_sampling)

    #This function returns the NN model for DQN
    #Weight Initializer: Glorot Uniform
    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.dynamic_state_size))) #number of observations
        model.add(Dense(256)) # Hidden Layer 1
        model.add(Activation('relu'))
        model.add(Dense(256)) # Hidden Layer 2
        model.add(Activation('relu'))
        model.add(Dense(128)) # Hidden Layer 3
        model.add(Activation('relu'))
        model.add(Dense(self.dynamic_action_size)) # Output Layer
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    #Loads level-k agent model in the given path
    def _load_levelk(self,path,random):
        agentlevk = DQNAgent(self.state_size, self.action_size) #Level-k Agent
        agentlevk.load(path) #Loads the model of trained level-k agent
        agentlevk.T = 1 #Sets the boltzmann temp. of Level-k cars to 1, prevents random actions
        agentlevk.boltzmann_sampling = random
        return agentlevk
    
    #Adds given experience to the experience set
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #Updates the target model by making its weights equal to the original model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_average_layer_weights(self):
        ave_weight = []
        ave_bias = []
        for idx_layer, layer in enumerate(self.model.layers):
            if idx_layer % 2 == 0:
                continue
            ave_weight.append(layer.get_weights()[0].mean())
            ave_bias.append(layer.get_weights()[1].mean())

        return ave_weight, ave_bias
    
    #Chooses a level-k strategy
    def act(self, state, remove_merging = False, get_qvals=False):

        if self.random_dynamic_strategy:
            random_strategy = random.randint(0,2)
            if get_qvals:
                levk_qvals = [self.agentlevk[0].act(state=state[:,:,0:self.state_size],
                                      remove_merging=remove_merging,
                                      get_qvals=get_qvals),
                self.agentlevk[1].act(state=state[:,:,0:self.state_size],
                                      remove_merging=remove_merging,
                                      get_qvals=get_qvals),
                self.agentlevk[2].act(state=state[:,:,0:self.state_size],
                                      remove_merging=remove_merging,
                                      get_qvals=get_qvals)]
                return [ ["Random-Strategy",random_strategy], levk_qvals]
            else:
                return [ ["Random-Strategy",random_strategy], self.agentlevk[random_strategy].act(
                    state=state[:,:,0:self.state_size], remove_merging=remove_merging)]
        else:
            q_values = self.model.predict(state).astype('float64')[0]
            greedy = np.argmax(q_values)
            exp_values = np.exp(q_values/self.T).astype('float64')
            rand = np.random.rand()
            total = np.float32(0)
            
            if get_qvals: # Return Q values with action related information
                
                levk_qvals = [self.agentlevk[0].act(state=state[:,:,0:self.state_size],
                                                    remove_merging=remove_merging,
                                                    get_qvals=get_qvals),
                              self.agentlevk[1].act(state=state[:,:,0:self.state_size],
                                                    remove_merging=remove_merging,
                                                    get_qvals=get_qvals),
                              self.agentlevk[2].act(state=state[:,:,0:self.state_size],
                                                    remove_merging=remove_merging,
                                                    get_qvals=get_qvals)]
                
                for i in range(self.dynamic_action_size):
                    # If softmax outputs are zero or infinite valued,
                    # return the greedy action
                    if np.sum(exp_values) == 0 or np.sum(exp_values) == np.inf:
                        return [ ["Argmax-Inf", q_values.tolist(), greedy], 
                                levk_qvals]
                    # Iteratively sum exponential values for the random action check below
                    total += (exp_values[i]/np.sum(exp_values))
                    
                    # Take random action by Boltzmann Sampling if the random number passes
                    # the current sum of softmax q-values
                    if self.boltzmann_sampling and (rand < total):
                        if greedy == i:
                            return [ ["Random-Argmax",q_values.tolist(),greedy],
                                    levk_qvals]
                        else:
                            return [ ["Random",q_values.tolist(),i], 
                                    levk_qvals]
                # If none of the above, return the greedy action
                return [ ["Argmax",q_values.tolist(),greedy], 
                        levk_qvals]
            else: # Return only action related information
                for i in range(self.dynamic_action_size):
                    # If softmax outputs are zero or infinite valued,
                    # return the greedy action
                    if np.sum(exp_values) == 0 or np.sum(exp_values) == np.inf:
                        return [ ["Argmax-Inf",greedy], 
                                self.agentlevk[greedy].act(state=state[:,:,0:self.state_size],
                                                           remove_merging=remove_merging)]
                    # Iteratively sum exponential values for the random action check below
                    total += (exp_values[i]/np.sum(exp_values))
                    
                    # Take random action by Boltzmann Sampling if the random number passes
                    # the current sum of softmax q-values
                    if self.boltzmann_sampling and (rand < total):
                        if greedy == i:
                            return [ ["Random-Argmax",greedy], 
                                    self.agentlevk[greedy].act(state=state[:,:,0:self.state_size],
                                                               remove_merging=remove_merging)]
                        else:
                            return [ ["Random",i], 
                                    self.agentlevk[i].act(state=state[:,:,0:self.state_size],
                                                          remove_merging=remove_merging)]
                # If none of the above, return the greedy action
                return [ ["Argmax", greedy], 
                        self.agentlevk[greedy].act(state=state[:,:,0:self.state_size],
                                                   remove_merging=remove_merging)]
            
    # Returns actions and level-k strategies of multiple dynamic agents 
    # through Boltzmann Sampling
    # remove_merging parameter determines whether to
    # remove the "merge" action from the last layer
    def act_inbatch(self, states, remove_merging=False):
        if self.random_dynamic_strategy:
            acts = np.zeros((states.shape[0]))
            random_strategies = np.array(random.choices( [0,1,2], k=states.shape[0]))
            # Sample action for each dynamic agent according to its level-k selection
            l1 = np.where(random_strategies==0)[0]
            l2 = np.where(random_strategies==1)[0]
            l3 = np.where(random_strategies==2)[0]
            if l1.size != 0:
                 acts[l1] = self.agentlevk[0].act_inbatch(states[l1,:,0:self.state_size],
                                                          remove_merging)
            if l2.size != 0:
                acts[l2] = self.agentlevk[1].act_inbatch(states[l2,:,0:self.state_size],
                                                         remove_merging)
            if l3.size != 0:
                 acts[l3] = self.agentlevk[2].act_inbatch(states[l3,:,0:self.state_size],
                                                          remove_merging)
            return acts, random_strategies

        else:
            Y_act = self.model.predict(states,batch_size=states.shape[0])

            select_levk = np.zeros((Y_act.shape[0]))
            acts = np.zeros((Y_act.shape[0]))
    
            if self.boltzmann_sampling: # Implement Boltzmann Sampling to select level-k strategies
                Y = np.exp(Y_act/self.T).astype('float64')
                Y_sum = np.sum(Y,axis=1)[:,None]
                
                # Check which Y_sum entries are NaN
                check = ((Y_sum == np.full(Y_sum.shape, np.inf)) | (Y_sum == np.full(Y_sum.shape, 0))).flatten()
                NaN = np.where(check==1)
                not_NaN = np.where(check==0)
                
                # For Y_sum entries with non-NaN values, implement Boltzmann sampling
                Y = Y[not_NaN] 
                Y_sum = Y_sum[not_NaN] 
                Y = Y/Y_sum
                rand = np.random.rand(Y.shape[0],1)
                total = np.zeros_like(Y).astype('float64')           
                for i in range(self.dynamic_action_size):
                    total[:,i] = np.sum(Y[:,0:i+1],axis=1)
                select_levk[not_NaN] = np.argmax(total>rand,axis=1)
                
                # For Y_sum entries with NaN values, implement greedy policy on 
                # the original q-values
                select_levk[NaN] = np.argmax(Y_act[NaN],axis=1)
                # Sample action for each dynamic agent according to its level-k selection
                l1 = np.where(select_levk==0)[0]
                l2 = np.where(select_levk==1)[0]
                l3 = np.where(select_levk==2)[0]
                if l1.size != 0:
                     acts[l1] = self.agentlevk[0].act_inbatch(states[l1,:,0:self.state_size],
                                                              remove_merging)
                if l2.size != 0:
                    acts[l2] = self.agentlevk[1].act_inbatch(states[l2,:,0:self.state_size],
                                                             remove_merging)
                if l3.size != 0:
                     acts[l3] = self.agentlevk[2].act_inbatch(states[l3,:,0:self.state_size],
                                                              remove_merging)
                return acts, select_levk
            else: 
                # Implement greedy policy for selecting level-k behaviour
                select_levk = np.argmax(Y_act,axis=1) 
                l1 = np.where(select_levk==0)[0]
                l2 = np.where(select_levk==1)[0]
                l3 = np.where(select_levk==2)[0]
                if l1.size != 0:
                     acts[l1] = self.agentlevk[0].act_inbatch(states[l1,:,0:self.state_size],
                                                              remove_merging)
                if l2.size != 0:
                    acts[l2] = self.agentlevk[1].act_inbatch(states[l2,:,0:self.state_size],
                                                             remove_merging)
                if l3.size != 0:
                     acts[l3] = self.agentlevk[2].act_inbatch(states[l3,:,0:self.state_size],
                                                              remove_merging)
                return acts, select_levk             

    #Experience replay
    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = np.array(random.sample(self.memory, batch_size), dtype= "object")
       
        X = np.squeeze(np.stack(minibatch[:,0]),axis=1)
        Y = self.model.predict(X, batch_size=batch_size)
        
        # Rewards of past experiences
        R = np.reshape(np.stack(minibatch[:,2]),[batch_size,1])
        
        # Q values of past experiences obtained from the target model
        Q_val = (np.max(self.target_model.predict(
            np.squeeze(np.stack(minibatch[:,3]),axis=1),
            batch_size=batch_size),axis=1)).reshape(-1,1)
        
        # Information about the status of the car, namely whether
        # the car passed the final point or crashed
        done = np.reshape(np.stack(minibatch[:,-1]),[batch_size,1])
        
        # Predicted Q-values
        Y[np.arange(batch_size)[:,None],
          np.reshape(np.stack(minibatch[:,1]),
                     [batch_size,1])] = R + self.gamma*np.multiply(Q_val,~done)

        # Fit the current model wrt the target model output
        training_hist = self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        gc.collect()
        tf.keras.backend.clear_session()
        #upper code segment was aimed to avoid memory leaks. It achieves less RAM usage. Yet, also leads to drop in GPU use!



        return training_hist.history['loss']

    def update_temperature(self, step):
        self.T = np.maximum(self.T - (self.MAX_T-self.MIN_T)*step, self.MIN_T) 

    def load_memory(self, fname):
        with open(fname, "rb") as input_file:
            self.memory = pickle.load(input_file)
    
    def save_memory(self,fname):
        with open(fname, "wb") as output_file:
            pickle.dump(self.memory, output_file)
            
    #Loads the Boltzmann temperature and total timesteps
    def load_config(self, config_fname):
        with open(config_fname, "rb") as input_file:
            # [self.T,self.epsilon, total_timesteps] = pickle.load(input_file)
            [self.T, total_timesteps] = pickle.load(input_file)
        return total_timesteps
    
    #Saves the last Boltzmann temperature and total timesteps
    def save_config(self, total_timesteps, config_fname):
        with open(config_fname, "wb") as output_file:
            # pickle.dump([self.T,self.epsilon, total_timesteps], output_file)
            pickle.dump([self.T, total_timesteps], output_file)
            
    #Loads the model, and the target weights for retraining
    def load(self, fname, tfname="", retrain=False):
        self.model = load_model(fname)
        if retrain:
            self.target_model.load_weights(tfname) 
            
    #Saves the model, and the target weights for retraining
    def save(self, fname, tfname="", backup = False):
        self.model.save(fname)
        if backup:
            self.target_model.save_weights(tfname)

        
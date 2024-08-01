import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque

class TrainModel:
    

    def __init__(self, num_states, num_actions, reward_table, learning_rate, discount_rate, epsilon, epsilon_decay, epsilon_min, max_episodes, max_steps_per_episode):
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_table = reward_table
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 48
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.experience_replay_capacity = 10000

        self.best_total_reward = float('-inf')
        self.best_model_weights = None
        self.experience_replay_buffer = deque(maxlen=self.experience_replay_capacity)
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=(self.num_states,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_model(self):
        if len(self.experience_replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.experience_replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        q_values_next = self.target_model.predict(next_states)
        q_values = self.model.predict(states)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target = rewards[i] + self.discount_rate * np.max(q_values_next[i])
            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, epochs=1, verbose=0)

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def train_main(self):
        all_episode_rewards = []
        for episode in range(self.max_episodes):
            state = np.zeros(self.num_states)
            state_index = 0
            total_reward = 0

            for step in range(self.max_steps_per_episode):
                action = self.choose_action(state, self.epsilon)
                next_state_index = state_index

                if action == 0 and state_index % 12 != 11:
                    next_state_index = state_index + 1
                elif action == 1 and state_index % 12 != 0:
                    next_state_index = state_index - 1
                elif action == 2 and state_index <= 35:
                    next_state_index = state_index + 12
                elif action == 3 and state_index >= 12:
                    next_state_index = state_index - 12

                reward = self.reward_table[next_state_index]
                done = reward == 100 or reward == -100
                total_reward += reward

                next_state = np.zeros(self.num_states)
                next_state[next_state_index] = 1

                self.experience_replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                state_index = next_state_index

                self.update_model()

                if done:
                    break

            all_episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

            if total_reward > self.best_total_reward and episode > 900:
                self.best_total_reward = total_reward
                self.best_model_weights = self.model.get_weights()

            if episode % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())

        if self.best_model_weights is not None:
            self.model.set_weights(self.best_model_weights)
            self.model.save_weights("best_model_weights.h5")

        plt.plot(all_episode_rewards)
        plt.title('Average Reward Per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.show()


def BestModel(num_states, num_actions, learning_rate):
    best_model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(num_states,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    best_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    best_model.load_weights('best_model_weights.h5')
    return best_model


num_states = 48
num_actions = 4
reward_table = np.zeros(num_states) - 1
reward_table[26:33] = -100
reward_table[17] = 100  # prize location
reward_table[20] = -100 # terminal state 1
reward_table[13] = -100 # terminal state 2

learning_rate = 0.2
discount_rate = 0.95
epsilon = 0.5
epsilon_decay = 0.95
epsilon_min = 0.01
max_episodes = 1000

trainer = TrainModel(num_states, num_actions, reward_table, learning_rate, discount_rate, epsilon, epsilon_decay, epsilon_min, max_episodes, 60)
trainer.train_main()

# Create the best model with the loaded weights
best_model = BestModel(num_states, num_actions, learning_rate)

"""From now on, play the game"""

states = list(range(48))
start_state = 0
all_game_rewards = []
rewards = []
states_check = []
action_check = []
reward_current = 0
reward = 0
epsilon_min = 1
my_rate = 1 - epsilon

for episode in range(1):
    state = np.zeros(num_states)  # Initialize state as one-hot vector
    state[start_state] = 0  # Set initial state
    reward_current = 0
    states_check.append(start_state)

    for step in range(40):
        lower = np.random.rand()
        done = False

        
        q_values = best_model.predict(state.reshape(1, -1))  # Ensure input shape is (1, num_states)
        action = np.argmax(q_values[0])
       
        action_check.append(action)

        new_state = np.zeros(num_states)  # New state as one-hot vector
        if action == 0:
            if state.argmax() % 12 != 11:
                new_state[state.argmax() + 1] = 1
        elif action == 1:
            if state.argmax() % 12 != 0:
                new_state[state.argmax() - 1] = 1
        elif action == 2:
            if state.argmax() <= 35:
                new_state[state.argmax() + 12] = 1
        elif action == 3:
            if state.argmax() >= 12:
                new_state[state.argmax() - 12] = 1
        states_check.append(new_state.argmax())

        reward = reward_table[new_state.argmax()]
        if reward == 100 or reward == -100:
            done = True
        reward_current += reward
        state = new_state  # Update state

        if done:
            break


    rewards.append(reward_current)
    done = False
 

plt.figure()
plt.plot( 500, rewards,  'o', label='Average Reward per Episode')
plt.title(f'Average Reward per Episode Across Games for Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()


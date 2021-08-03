import random
import numpy as np
from collections import deque

from agent.model import QTrainer, build_model

# memory
BATCH_SIZE = 32
MAX_MEM = 100

# hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.99
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

class Agent:
    def __init__(self, actions, states, file_name, train):
        self.epsilon = 1
        self.n_episodes = 0
        self.states = states
        self.actions = actions
        self.memory = deque(maxlen=MAX_MEM)
        self.model = build_model(self.states, actions, file_name, train, lr=LEARNING_RATE)
        self.trainer = QTrainer(model=self.model, lr=LEARNING_RATE, gamma=DISCOUNT_RATE)

    def preprocess_state(self, state):
        return np.reshape(state, (1, self.states))

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def get_actions(self, state, train):
        ac_count = self.actions[0].n

        actions = np.zeros((ac_count, 2), dtype="int16")

        if train and np.random.rand() <= self.epsilon:
            fake_prediction = np.random.random(size=(ac_count, 2))
            indices = np.argmax(fake_prediction, axis=1)

            for i in range(len(indices)): actions[i][indices[i]] = 1

            return actions
        else:
            prediction = self.model.predict(self.preprocess_state(state))
            indices = np.argmax(np.reshape(prediction, (ac_count, 2)), axis=1)
            
            for i in range(len(indices)): actions[i][indices[i]] = 1

            return actions

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))  # list of tuples

        for state, actions, reward, next_state, done in mini_sample:
            self.trainer.train_step(
                self.preprocess_state(state), 
                actions, 
                reward, 
                self.preprocess_state(next_state), 
                done
            )

    def train_short_memory(self, state, actions, reward, next_state, done):
        self.trainer.train_step(
            self.preprocess_state(state), 
            actions, 
            reward, 
            self.preprocess_state(next_state), 
            done
        )

    def decrease_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

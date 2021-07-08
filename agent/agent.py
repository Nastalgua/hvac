import random
import sys
import numpy as np
from collections import deque

from agent.model import QTrainer, build_model

# memory
BATCH_SIZE = 32
MAX_MEM = 2000

# hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.95
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1

class Agent:
    def __init__(self, actions, states, file_name, train):
        self.n_episodes = 0
        self.epsilon = 1  # randomness
        self.actions = actions
        self.memory = deque(maxlen=MAX_MEM)
        self.model = build_model(states, actions, file_name, train, lr=LEARNING_RATE)
        self.trainer = QTrainer(model=self.model, lr=LEARNING_RATE, gamma=DISCOUNT_RATE)

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def get_actions(self, state, train):
        if train and np.random.rand() <= self.epsilon:
            return np.random.randint(0, high=2, size=(1, 3), dtype="int32")
        else:
            prediction = self.model.predict(state)
            actions = np.zeros((1, int(prediction.shape[1] / 2)), dtype="int32")
            
            for i in range(len(prediction[0])):
                if i % 2 != 0: continue
                actions[0][int(i / 2)] = np.argmax([prediction[0][i], prediction[0][i + 1]])
                
            return actions

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return 0

        mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples

        for state, actions, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, actions, reward, next_state, done)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def train_short_memory(self, state, actions, reward, next_state, done):
        self.trainer.train_step(state, actions, reward, next_state, done)

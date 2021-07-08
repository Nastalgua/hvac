from env.hvac_env import AC_COUNT
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
        actions = np.zeros((AC_COUNT, 2), dtype="int16")

        if train and np.random.rand() <= self.epsilon:
            fake_precdiction = np.random.random(size=(AC_COUNT, 2))
            indices = np.argmax(fake_precdiction, axis=1)

            for i in range(len(indices)): actions[i][indices[i]] = 1

            return actions
        else:
            prediction = self.model.predict(state)
            indices = np.argmax(np.reshape(prediction, (AC_COUNT, 2)), axis=1)

            for i in range(len(indices)): actions[i][indices[i]] = 1
                
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

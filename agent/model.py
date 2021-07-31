import sys
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

def build_model(states, actions, file_name, train, lr):
    model = Sequential()
    
    model.add(Dense(24, activation="relu", input_shape=(states, )))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions[0].n * actions[1].n, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=lr, decay=0.01), loss='mse')

    if not train:
        model.load_weights("weights/" + file_name + ".h5")

    return model

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model

    def train_step(self, state, actions, reward, next_state, done):
        target = reward
        
        if not done:
            target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
        
        target_f = self.model.predict(state)

        for action in range(len(actions[0])):
            if actions[0][action] == 1:
                target_f[0][action] = target
        
        self.model.fit(state, target_f, verbose=0)

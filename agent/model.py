import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer

from tensorflow.keras.optimizers import Adam

def build_model(states, actions, file_name, train, lr, ac_count):
    model = Sequential()
    
    model.add(Flatten(input_shape=(states, )))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(2 * ac_count, activation='linear'))

    model.compile(Adam(learning_rate=lr, decay=0.01), 'mse')
    
    if not train:
        model.load_weights("weights/" + file_name + ".h5")

    return model

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model

    def train_step(self, state, actions, reward, next_state, done, ac_count):
        target = reward
        
        if not done:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, 3))[0]))

        target_f = self.model.predict(state.reshape(1, 3))
        actions = np.reshape(actions, (1, ac_count * 2))

        for action in range(len(actions[0])):
            target_f[0][action] = target
        
        self.model.fit(state.reshape(1, 3), np.array(target_f), verbose=0)

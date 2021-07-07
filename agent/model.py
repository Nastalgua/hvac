import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.optimizers import Adam

def build_model(states, actions, file_name, train, lr):
    model = Sequential()
    model.add(Flatten(input_shape=(states, )))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation='linear'))

    model.compile(Adam(learning_rate=lr), 'mse')
    
    if not train:
        model.load_weights("weights/" + file_name + ".h5")

    return model

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model

    def train_step(self, state, action, reward, next_state, done):
        target = reward

        if not done:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, verbose=0)


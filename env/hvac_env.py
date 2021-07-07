import numpy as np

from gym import Env
from gym.spaces import Discrete, Box
from numpy import random
from numpy.lib.scimath import sqrt

from ac import AC
from target import Target
from visual import Visual, convert_pos

MASK = np.matrix([1.0 / 9.0]).repeat(9).reshape(3, 3)
AC_COUNT = 3

def conv2d(input_matrix: np.ndarray, mask: np.ndarray):
    view_shape = mask.shape + tuple(np.subtract(input_matrix.shape, mask.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    sub_matrix = strd(input_matrix, shape=view_shape, strides=input_matrix.strides * 2)

    return np.einsum('ij,ijkl->kl', mask, sub_matrix)

class HvacEnv(Env):
    def __init__(self):
        # TODO: action space
        # Even numbers = off
        # odd numbers = on
        self.action_space = Discrete(AC_COUNT * 2)

        # TODO: observation space

        # start temps
        self.state = np.random.randint(100, size=(10, 10))
        self.acs: AC = np.array([
            AC((7, 3)),
            AC((2, 6)),
            AC((7, 7)),
        ], dtype=object)
        self.targets: Target = np.array([
            Target((3, 7)),
            Target((4, 3))
        ], dtype=object)
        
        # time
        self.apply_length = 60

        # gui init
        self.show_gui = False
        self.gui = Visual(self.state)

    def step(self, action):
        self.apply_length -= 1
        turnOn = action % 2 != 0

        if turnOn: # off action
            ac_index = int(action + 1) / 2
        else: # on action
            ac_index = int(action) / 2
        
        currentAC: AC = self.acs[int(ac_index)]
        currentAC.on = turnOn
        
        for ac in self.acs:
            # ac temp decrease
            self.state[ac.position[0]][ac.position[1]] += ac.factor
        
        self.state = conv2d(self.state, MASK)
        self.state = np.pad(self.state, 1, constant_values=[72])
    
        # calculate reward 
        for t in self.targets:
            t: Target = t
            currentPositionTemp = self.state[t.position[0]][t.position[1]]
            
            delta = abs(currentPositionTemp - 74)

            if delta < 0.4:
                reward = 200
            else:
                capped_delta = min(delta, sqrt(200))
                reward = capped_delta ** 2
        
        # calculate done 
        if self.apply_length <= 0:
            done = True
        else:
            done = False

        info = {}

        if (self.show_gui):
            self.gui.updateData(self.state)

        return self.state, reward, done, info

    def render(self):
        self.show_gui = True
        self.gui.render()

    def reset(self):
        self.state = np.random.randint(100, size=(10, 10))
        self.gui.updateData(self.state)
        return self.state

env = HvacEnv()

finished = False

while True:
    # testing step()
    if (not finished):
        new_state, reward, done, info = env.step(0)
        finished = done

    # testing render() and reset()
    if (random.randint(0, 100) == 50):
        env.reset()
    
    env.render()

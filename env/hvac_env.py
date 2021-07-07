import numpy as np

from gym import Env
from gym.spaces import Discrete, Box
from numpy import random
import math 

from ac import AC
from target import Target
from visual import Visual

MASK = np.matrix([1.0 / 9.0]).repeat(9).reshape(3, 3)
AC_COUNT = 3

def conv2d(input_matrix: np.ndarray, mask: np.ndarray):
    view_shape = mask.shape + tuple(np.subtract(input_matrix.shape, mask.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    sub_matrix = strd(input_matrix, shape=view_shape, strides=input_matrix.strides * 2)

    return np.einsum('ij,ijkl->kl', mask, sub_matrix)

class HvacEnv(Env):
    def __init__(self):
        # Even numbers = off
        # odd numbers = on
        self.action_space = Discrete(AC_COUNT * 2)

        self.observation_space = Box(
            low=np.array([0, 0, 0], dtype=np.float32), 
            high=np.array([100, 100, 100], dtype=np.float32)
        )

        # start temps
        self.grid = np.random.randint(100, size=(8, 8))
        self.grid = np.pad(self.grid, 1, constant_values=[72])

        self.acs = np.array([
            AC((7, 3)),
            AC((2, 6)),
            AC((7, 7)),
        ], dtype=object)
        
        self.targets = np.array([
            Target((3, 7)),
            Target((4, 3))
        ], dtype=object)

        self.state = np.array([0, 0])
        
        # time
        self.apply_length = 60

        # gui init
        self.show_gui = False
        self.gui = Visual(self.grid)

    def step(self, action):
        self.apply_length -= 1

        turnOn = action % 2 != 0
        ac_index = int(action) / 2

        currentAC: AC = self.acs[int(ac_index)]
        currentAC.on = turnOn
        
        for ac in self.acs:
            # ac temp decrease
            self.grid[ac.position[0]][ac.position[1]] += ac.factor
        
        prev_grid = self.grid

        self.grid = conv2d(self.grid, MASK)
        self.grid = np.pad(self.grid, 1, constant_values=[72])

        for ac in self.acs:
            # ac temp decrease
            self.grid[ac.position[0]][ac.position[1]] = prev_grid[ac.position[0]][ac.position[1]]

        # calculate reward 
        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            currentPositionTemp = self.grid[t.position[0]][t.position[1]]
            self.state[i] = currentPositionTemp

            delta = abs(currentPositionTemp - 74)

            if delta < 0.4:
                reward = 200
            else:
                capped_delta = min(delta, math.sqrt(200))
                reward = capped_delta ** 2
        
        # calculate done 
        if self.apply_length <= 0:
            done = True
        else:
            done = False

        info = {}

        if (self.show_gui):
            self.gui.updateData(self.grid)

        return self.state, reward, done, info

    def render(self):
        self.show_gui = True
        self.gui.render()

    def reset(self):
        self.grid = np.random.randint(100, size=(10, 10))
        self.gui.updateData(self.grid)
        self.apply_length = 60
        return self.grid

env = HvacEnv()

episode = 0

while True:
    episode += 1
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    
    print('Episode:{} Score:{}'.format(episode, score))

'''
finished = False

while True:
    # testing step()
    if (not finished):
        new_state, reward, done, info = env.step(0)
        finished = done

    # testing render() and reset()
    if (random.randint(0, 100) == 50):
        env.reset()
        finished = False
    
    env.render()
'''

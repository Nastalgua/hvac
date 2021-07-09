import os
import math 
import numpy as np, random

from gym import Env
from gym.spaces import MultiDiscrete, Box

from type_models.ac import AC
from type_models.wall import Wall
from type_models.target import Target
from env.visual import Visual

MASK = np.matrix([1.0 / 9.0]).repeat(9).reshape(3, 3)

AC_COUNT = 3
AC_FIXED_TEMP = 55
OUTSIDE_TEMP = 90

def conv2d(input_matrix: np.ndarray, mask: np.ndarray):
    view_shape = mask.shape + tuple(np.subtract(input_matrix.shape, mask.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    sub_matrix = strd(input_matrix, shape=view_shape, strides=input_matrix.strides * 2)

    return np.einsum('ij,ijkl->kl', mask, sub_matrix)

class HvacEnv(Env):
    def __init__(self):
        # 0 = off, 1 = on
        self.action_space = MultiDiscrete(np.full((AC_COUNT, 2), 2, dtype='int16'))

        self.observation_space = Box(
            low=np.array([0, 0], dtype='float32'), 
            high=np.array([100, 100], dtype='float32')
        )
        
        self.walls = np.array([], dtype=object)
        self.targets = np.array([], dtype=object)
        self.acs = np.array([], dtype=object)

        self.state = np.array([], dtype='int64')

        # process map.txt file
        cwd = os.getcwd()
        col = 0; max_length = 0; max_width = 0
        with open(cwd + "\\env\\map.txt") as f:
            for i in f.readlines():
                line = i.replace(" ", "")
                print(line)
                for row in range(0, len(line)):
                    max_width = max(row, max_width)

                    if line[row] == '*': # wall
                        self.walls = np.append(self.walls, Wall((row + 1, col + 1)))
                    elif line[row] == 'T':
                        self.targets = np.append(self.targets, Target((row + 1, col + 1)))
                        self.state = np.append(self.state, 0)
                    elif line[row] == 'A':
                        self.acs = np.append(self.acs, AC((row + 1, col + 1)))
            
                col += 1
                max_length = max(row, max_length)

        # start temps
        self.grid = np.random.randint(100, size=(max_width, max_length))
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])
        for ac in self.acs: self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP
        
        # time
        self.apply_length = 2400

        # gui init
        self.show_gui = False
        self.gui = Visual(self.grid, self.walls, self.acs, self.targets)

    def step(self, actions):
        self.apply_length -= 1

        # action processing
        for i in range(len(actions)):
            self.acs[i].on = (actions[i][1] == 1)
        
        # apply mask
        self.grid = conv2d(self.grid, MASK)
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])

        for ac in self.acs:
            if ac.on:
                self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP

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
                reward = 200 - (capped_delta ** 2)
        
        # calculate done 
        if self.apply_length <= 0:
            done = True
        else:
            done = False

        info = {}

        if (self.show_gui): self.gui.updateData(self.grid) # gui

        return self.state, reward, done, info

    def render(self):
        self.show_gui = True
        self.gui.render()

    def reset(self):
        self.grid = np.random.randint(100, size=(10, 10))
        self.gui.updateData(self.grid)
        self.apply_length = 60

        return self.state

'''
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
'''
env = HvacEnv()

finished = False

while True:    
    env.render()
'''

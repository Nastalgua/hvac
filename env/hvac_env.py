import os
import math 
import numpy as np

from gym import Env
from gym.spaces import MultiDiscrete, Box

from env.type_models.ac import AC
from env.type_models.wall import Wall
from env.type_models.target import Target

from env.visual import Visual
from env.conductivity import OUTSIDE_TEMP, apply_conductivity

APPLY_LENGTH = 60

TARGET_TEMP = 74
AC_FIXED_TEMP = 55

MAP_FILE_NAME = "map.txt";

class HvacEnv(Env):
    def __init__(self):
        self.observation_space = Box(
            low=np.array([0, 0], dtype='float32'), 
            high=np.array([100, 100], dtype='float32')
        )
        
        self.walls = np.array([], dtype=object)
        self.targets = np.array([], dtype=object)
        self.acs = np.array([], dtype=object)
        
        self.state = np.array([], dtype='int64')

        self.ac_count = 0

        # process map
        cwd = os.getcwd()
        with open(cwd + "\\env\\maps\\" + MAP_FILE_NAME) as f: # get length and height
            j = 0        
            for line in f.readlines():
                self.max_width = math.ceil(len(line) / 2.0)
                j += 1
            
            self.max_height = j
            f.close()
        
        self.conductivity = np.zeros((self.max_width, self.max_height))

        col = 0;
        with open(cwd + "\\env\\maps\\" + MAP_FILE_NAME) as f:
            for i in f.readlines():
                line = i.replace(" ", "")

                for row in range(0, len(line)):
                    if line[row] == '*': # wall
                        self.walls = np.append(self.walls, Wall((row + 1, col + 1)))
                        self.conductivity[row, col] = 0.2
                    elif line[row] == 'T': # target
                        self.targets = np.append(self.targets, Target((row + 1, col + 1)))
                        self.state = np.append(self.state, 0)
                        self.conductivity[row, col] = 1
                    elif line[row] == 'A': # AC
                        self.ac_count += 1
                        self.acs = np.append(self.acs, AC((row + 1, col + 1)))
                        self.conductivity[row, col] = 1
                    elif line[row] == 'D': # door
                        self.conductivity[row, col] = 0.5
                    elif line[row] == 'I': # door
                        self.conductivity[row, col] = 0.3
                    elif line[row] == '_':
                        self.conductivity[row, col] = 1
            
                col += 1
            f.close()

        self.action_space = MultiDiscrete(np.full((self.ac_count, 2), 2, dtype='int16'))

        # start temps
        self.grid = np.random.randint(100, size=(self.max_width, self.max_height))
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])
        for ac in self.acs: self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP

        # time
        self.apply_length = APPLY_LENGTH

        # gui init
        self.show_gui = False
        self.gui = Visual(self.grid, self.walls, self.acs, self.targets)

    def step(self, actions):
        self.apply_length -= 1

        # process action
        if len(self.acs) > 0:
            for i in range(len(actions)):
                self.acs[i].on = (actions[i, 1] == 1)

        self.grid = apply_conductivity(self.grid, self.conductivity)
        
        # maintain AC temp
        for ac in self.acs:
            if ac.on:
                self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP

        # calculate reward 
        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            current_pos_temp = self.grid[t.position[0]][t.position[1]]
            self.state[i] = current_pos_temp

            delta = abs(current_pos_temp - TARGET_TEMP)

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
        self.grid = np.random.randint(100, size=(self.max_width, self.max_height))
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])
        
        for ac in self.acs: self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP

        self.gui.updateData(self.grid)
        self.apply_length = APPLY_LENGTH

        return self.state

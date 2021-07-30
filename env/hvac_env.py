import os
import math
import random
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box, Tuple

from env.type_models.ac import AC
from env.type_models.wall import Wall
from env.type_models.target import Target

from env.visual import Visual
from env.conductivity import OUTSIDE_TEMP, apply_conductivity

APPLY_LENGTH = 400

START_TEMP = 75
AC_FIXED_TEMP = 55

LOW_TEMP_RANGE = 73
HIGH_TEMP_RANGE = 77

WALL_CONDUCTIVITY = 0.01
TARGET_CONDUCTIVITY = 1
AC_CONDUCTIVITY = 1
DOOR_CONDUCTIVITY = 0.5
INNER_WALL_CONDUCTIVITY = 0.3
EMPTY_SPACE_CONDUCTIVITY = 1

MAP_FILE_NAME = "map.txt";

class HvacEnv(Env):
    def __init__(self):    
        self.walls = np.array([], dtype=object)
        self.targets = np.array([], dtype=object)
        self.acs = np.array([], dtype=object)
        
        self.state = np.array([], dtype='float32')

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
                        self.conductivity[row, col] = WALL_CONDUCTIVITY
                    elif line[row] == 'T': # target
                        self.targets = np.append(self.targets, Target((row + 1, col + 1)))
                        self.conductivity[row, col] = TARGET_CONDUCTIVITY
                    elif line[row] == 'A': # AC
                        self.ac_count += 1
                        self.acs = np.append(self.acs, AC((row + 1, col + 1)))
                        self.conductivity[row, col] = AC_CONDUCTIVITY
                    elif line[row] == 'D': # door
                        self.conductivity[row, col] = DOOR_CONDUCTIVITY
                    elif line[row] == 'I': # inner wall
                        self.conductivity[row, col] = INNER_WALL_CONDUCTIVITY
                    elif line[row] == '_': # empty space
                        self.conductivity[row, col] = EMPTY_SPACE_CONDUCTIVITY
            
                col += 1
            f.close()

        # create observation space
        high = np.array([
            OUTSIDE_TEMP - 76 # max differece
        ], dtype=np.float32)

        low = np.array([
            AC_FIXED_TEMP - 69 # min differece
        ], dtype=np.float32)

        self.observation_space = Box(
            low=low, 
            high=high,
            dtype=np.float32
        )
        
        self.action_space = Tuple([Discrete(self.ac_count), Discrete(2)])

        # set target temperature
        self.set_target_temp()

        # set grid temps
        self.grid = np.full((self.max_width, self.max_height), START_TEMP)
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])

        # set state
        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            current_temp = self.grid[t.position[0]][t.position[1]]
            self.state = np.append(self.state, [current_temp - self.target_temp])
        # current_temp, self.target_temp, 
        
        self.old_delta = -1

        # time
        self.apply_length = APPLY_LENGTH

        # gui init
        self.show_gui = False
        self.gui = Visual(self.grid, self.walls, self.acs, self.targets)

    def step(self, actions):
        if self.grid[self.targets[0].position[0]][self.targets[0].position[1]] < self.target_temp:
            if actions[0, 0] == 1:
                print('\033[94m' + 'Passed!' + '\033[0m')
            else:
                print('\033[91m' + 'Failed.' + '\033[0m')
        else:
            if actions[0, 1] == 1:
                print('\033[94m' + 'Passed!' + '\033[0m')
            else:
                print('\033[91m' + 'Failed.' + '\033[0m')

        self.apply_length -= 1

        # process action
        if len(self.acs) > 0:
            for i in range(len(actions)):
                '''
                # "Dumb" controller
                t: Target = self.targets[0]

                current_pos_temp = self.grid[t.position[0]][t.position[1]]

                if current_pos_temp > self.target_temp:
                    self.acs[i].on = True
                else:
                    self.acs[i].on = False
                '''
                # "smart" controller
                self.acs[i].on = (actions[i, 1] == 1)
        
        self.grid = apply_conductivity(self.grid, self.conductivity)
        
        for ac in self.acs:
            if ac.on:
                self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP
        
        self.state = np.array([], dtype='float32')

        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            current_temp = self.grid[t.position[0]][t.position[1]]
            self.state = np.append(self.state, [current_temp - self.target_temp])

        # calculate done 
        done = False
        if self.apply_length <= 0:
            done = True

        # calculate reward 
        total_reward = 0
        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            current_pos_temp = self.grid[t.position[0]][t.position[1]]
            
            delta = abs(current_pos_temp - self.target_temp)

            if delta < 0.5:
                print('success')
                total_reward += 50
                done = True
            # else:
            #     capped_delta = min(delta, math.sqrt(9))
            #     total_reward = 9 - (capped_delta ** 2)
            #     total_reward = 0
                # total_reward = 201 - min(delta * 15, 201)
            
            if self.old_delta != -1:
                if self.old_delta > delta:
                    total_reward -= 2
                else:
                    total_reward += 2

            self.old_delta = delta

        total_reward -= APPLY_LENGTH - self.apply_length

        info = {}

        if (self.show_gui): self.gui.updateData(self.grid) # gui

        return self.state, total_reward, done, info

    def render(self):
        self.show_gui = True
        self.gui.render()

    def reset(self):
        # set target temperature
        self.set_target_temp()
        # print(self.target_temp)
        
        # reset grid
        self.grid = np.full((self.max_width, self.max_height), START_TEMP)
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])

        # reset length
        self.apply_length = APPLY_LENGTH

        self.state = np.array([], dtype='float32')

        self.old_delta = -1
        
        # set state
        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            current_temp = self.grid[t.position[0]][t.position[1]]
            self.state = np.append(self.state, [current_temp - self.target_temp])

        return self.state
    
    def target_temps(self):
        temps = np.ndarray(self.targets.shape, dtype='float32')

        for i in range(len(self.targets)):
            t: Target = self.targets[i]
            temps[i] = self.grid[t.position[0], t.position[1]]

        return temps
    
    def set_target_temp(self):
        self.target_temp = random.randint(LOW_TEMP_RANGE, HIGH_TEMP_RANGE)

        while self.target_temp == START_TEMP:
            self.target_temp = random.randint(LOW_TEMP_RANGE, HIGH_TEMP_RANGE)

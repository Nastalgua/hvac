import os
import math
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box, Tuple

from env.type_models.ac import AC
from env.type_models.wall import Wall
from env.type_models.thermostat import Thermostat

from env.visual import Visual
from env.schedule.scheduler import Scheduler
from env.conductivity import OUTSIDE_TEMP, apply_conductivity

APPLY_LENGTH = 400

START_TEMP = 75
AC_FIXED_TEMP = 55

LOW_TEMP_RANGE = 73
HIGH_TEMP_RANGE = 77

WALL_CONDUCTIVITY = 0.01
THERMOSTAT_CONDUCTIVITY = 1
AC_CONDUCTIVITY = 1
DOOR_CONDUCTIVITY = 0.5
INNER_WALL_CONDUCTIVITY = 0.3
EMPTY_SPACE_CONDUCTIVITY = 1

MAP_FILE_NAME = "map.txt"

PASS_COLOR = '\033[94m'
FAIL_COLOR = '\033[91m'
SOLVE_SUCCESS_COLOR = '\033[92m'
RESET_COLOR = '\033[0m'

class HvacEnv(Env):
    def __init__(self):
        self.walls = np.array([], dtype=object)
        self.thermostats = np.array([], dtype=object)
        self.acs = np.array([], dtype=object)

        self.process_map()

        self.scheduler = Scheduler(states=len(self.thermostats))

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

        # set grid temps
        self.grid = np.full((self.max_width, self.max_height), START_TEMP)
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])

        # set state
        self.state = np.array([], dtype=np.float32)

        # keep track of total steps
        self.step_count = 0

        # keep track of old changes
        self.old_deltas = np.full((1, len(self.thermostats)), -1, dtype=np.float32)

        # time (prevent AI from creating a infinite loop)
        self.apply_length = APPLY_LENGTH

        # gui init
        self.show_gui = False
        self.gui = Visual(self.grid, self.walls, self.acs, self.thermostats)

    def step(self, actions):
        # determine if AI is on right course
        if self.grid[self.thermostats[0].position[0]][self.thermostats[0].position[1]] < self.target_temp:
            if actions[0, 0] == 1:
                print(PASS_COLOR + 'Passed!' + RESET_COLOR)
            else:
                print(FAIL_COLOR + 'Failed.' + RESET_COLOR)
        else:
            if actions[0, 1] == 1:
                print(PASS_COLOR + 'Passed!' + RESET_COLOR)
            else:
                print(FAIL_COLOR + 'Failed.' + RESET_COLOR)

        self.step_count += 1
        self.apply_length -= 1

        # process action
        if len(self.acs) > 0:
            for i in range(len(actions)): # len(actions) == len(acs)
                current_ac: AC = self.acs[i]

                current_ac.on = (actions[i, 1] == 1)
                
                if current_ac.on:
                    self.grid[current_ac.position[0]][current_ac.position[1]] = AC_FIXED_TEMP

        # apply the temperature conductivity
        self.grid = apply_conductivity(self.grid, self.conductivity)
        
        # maintain AC temperature after applying conductivity
        for ac in self.acs:
            if ac.on:
                self.grid[ac.position[0]][ac.position[1]] = AC_FIXED_TEMP
        
        # update state
        self.state = np.array([], dtype=np.float32)

        for i in range(len(self.thermostats)):
            t: Thermostat = self.thermostats[i]
            current_temp = self.grid[t.position[0]][t.position[1]]

            self.state = np.append(self.state, [current_temp - self.target_temp])
        
        self.scheduler.save_differences(self.state)

        # calculate done 
        done = False
        if self.apply_length <= 0:
            done = True

        # calculate reward 
        reward = 0
        for i in range(len(self.thermostats)): # len(self.thermostats) == len(self.deltas)
            t: Thermostat = self.thermostats[i]
            current_pos_temp = self.grid[t.position[0]][t.position[1]]
            
            delta = abs(current_pos_temp - self.target_temp)

            if delta < 0.5:
                print(SOLVE_SUCCESS_COLOR + 'Success!' + RESET_COLOR)
                reward += 50
                done = True
            
            if self.old_deltas[0][i] != -1:
                if self.old_deltas[0][i] > delta: # moving away from the target number
                    reward -= 2
                else:
                    reward += 2

            self.old_deltas[0][i] = delta
        
        # discourage AI from taking too long
        reward -= (APPLY_LENGTH - self.apply_length)

        info = {}

        if (self.show_gui): self.gui.updateData(self.grid) # gui

        return self.state, reward, done, info

    def render(self):
        self.show_gui = True
        self.gui.render()

    def reset(self):      
        self.grid = np.full((self.max_width, self.max_height), START_TEMP)
        self.grid = np.pad(self.grid, 1, constant_values=[OUTSIDE_TEMP])

        self.apply_length = APPLY_LENGTH

        self.old_deltas = np.full((1, len(self.thermostats)), -1, dtype=np.float32)

        self.state = np.array([], dtype=np.float32)

        self.set_target_temp()

        for i in range(len(self.thermostats)):
            t: Thermostat = self.thermostats[i]
            current_temp = self.grid[t.position[0]][t.position[1]]

            self.state = np.append(
                self.state, 
                [current_temp - self.target_temp]
            )
        
        self.scheduler.save_differences(self.state)

        return self.state
    
    def get_thermostat_temps(self):
        temps = np.ndarray(self.thermostats.shape, dtype=np.float32)

        for i in range(len(self.thermostats)):
            t: Thermostat = self.thermostats[i]
            temps[i] = self.grid[t.position[0], t.position[1]]

        return temps

    def set_target_temp(self):
        self.target_temp = self.scheduler.get_target_temp(step_count=self.step_count)

    def process_map(self):
        self.ac_count = 0

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
                    elif line[row] == 'T': # thermostat
                        self.thermostats = np.append(self.thermostats, Thermostat((row + 1, col + 1)))
                        self.conductivity[row, col] = THERMOSTAT_CONDUCTIVITY
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
                
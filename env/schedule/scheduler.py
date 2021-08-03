import os
import numpy as np

SET_TARGET_TEMP_MAX_STEP_COUNT = 60
SET_OUTSIDE_TEMP_MAX_STEP_COUNT = 120

TARGET_TEMPS_FILE_NAME = "target_temps.txt"
OUTSIDE_TEMPS_FILE_NAME = "outside_temps.txt"

class Scheduler:
    def __init__(self, states):
        self.target_temps = np.array([], dtype=np.float32)
        self.outside_temps = np.array([], dtype=np.float32)
        
        self.outside_temp_index = 0
        self.target_temp_index = 0
        self.states = states

        self.difference_history = np.zeros((1, self.states), dtype=np.float32)
        self.difference_sums = np.zeros((1, self.states), dtype=np.float32)
        
        self.process_target_temps()
        self.process_outside_temps()

        self.curr_outside_temp = self.outside_temps[self.outside_temp_index]
        self.max_outside_temp = np.amax(self.curr_outside_temp)
    
    def get_target_temp(self, step_count):
        target_temp = self.target_temps[self.target_temp_index]

        if step_count != 0 and step_count % SET_TARGET_TEMP_MAX_STEP_COUNT == 0:
            self.target_temp_index += 1

            target_temp = self.target_temps[self.target_temp_index]

            # calc the sum
            temp_sums = np.array([], dtype=np.float32)
            for i in range(len(self.difference_history[0])):
                difference_history_column = self.get_history_column(i)
                temp_sums = np.append(temp_sums, [self.sum(difference_history_column)])

            reshaped_temp_sums = np.reshape(temp_sums, (1, self.states))
            self.difference_sums = np.append(self.difference_sums, reshaped_temp_sums, axis=0)

            if self.target_temp_index == len(self.target_temps) - 1:
                self.target_temp_index = 0;

        return target_temp
    
    def set_curr_outside_temp(self, step_count):
        if step_count != 0 and step_count % SET_OUTSIDE_TEMP_MAX_STEP_COUNT == 0:
            self.outside_temp_index += 1
            self.curr_outside_temp = self.outside_temps[self.outside_temp_index]

            if self.outside_temp_index == len(self.outside_temps) - 1:
                self.outside_temp_index = 0;

    def get_history_column(self, index):
        return self.difference_history[:, index][1:]
    
    def get_sum_column(self, index):
        return self.difference_sums[:, index][1:]

    def save_differences(self, differences):
        reshaped_differences = np.reshape(differences, (1, self.states))
        self.difference_history = np.append(self.difference_history, reshaped_differences, axis=0)

    def reset_differences(self):
        self.difference_history = np.zeros((1, self.states), dtype=np.float32)

    def sum(self, differences):
        return np.sum(differences)

    def process_target_temps(self):
        cwd = os.getcwd()

        with open(cwd + "\\env\\schedule\\" + TARGET_TEMPS_FILE_NAME) as f:
            for i in f.readlines():
                num = int(i.replace(" ", ""))
                self.target_temps = np.append(self.target_temps, [num])
    
    def process_outside_temps(self):
        cwd = os.getcwd()

        with open(cwd + "\\env\\schedule\\" + OUTSIDE_TEMPS_FILE_NAME) as f:
            for i in f.readlines():
                num = float(i.replace("\n", ""))
                self.outside_temps = np.append(self.outside_temps, [num])
    
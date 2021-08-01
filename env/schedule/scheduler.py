import os
import numpy as np

SET_TEMP_MAX_STEP_COUNT = 200

TARGET_TEMPS_FILE_NAME = "target_temps.txt"

class Scheduler:
    def __init__(self, states):
        self.schedule = np.array([], dtype=np.float32)
        self.target_temp_counter = 0
        self.states = states

        self.difference_history = np.zeros((1, self.states), dtype=np.float32)
        self.difference_sums = np.zeros((1, self.states), dtype=np.float32)
        
        self.process_schedule()
    
    def get_target_temp(self, step_count):
        target_temp = self.schedule[self.target_temp_counter]

        if step_count != 0 and step_count % SET_TEMP_MAX_STEP_COUNT == 0:
            self.target_temp_counter += 1
            target_temp = self.schedule[self.target_temp_counter]

            print('get next target temp')

            # calc the sum
            temp_sums = np.array([], dtype=np.float32)
            for i in range(len(self.difference_history[0])):
                difference_history_column = self.get_history_column(i)
                temp_sums = np.append(temp_sums, [self.sum(difference_history_column)])

            reshaped_temp_sums = np.reshape(temp_sums, (1, self.states))
            self.difference_sums = np.append(self.difference_sums, reshaped_temp_sums, axis=0)

        return target_temp

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

    def process_schedule(self):
        cwd = os.getcwd()

        with open(cwd + "\\env\\schedule\\" + TARGET_TEMPS_FILE_NAME) as f:
            for i in f.readlines():
                num = int(i.replace(" ", ""))
                self.schedule = np.append(self.schedule, [num])
    
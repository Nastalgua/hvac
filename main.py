import csv
import random

import numpy as np

from matplotlib import pyplot as plt

from agent.agent import Agent

from env.hvac_env import HvacEnv

N_EPISODES = 250
MAX_TIMESTEP = 1000

episode_endpoints = np.array([], dtype=np.float32)
rewards = np.array([], dtype=np.float32)
times = np.array([], dtype=np.int32)
target_temps = np.array([], dtype=np.float32)
difference_sums = np.array([], dtype=np.float32)

target_fs = np.array([], dtype=np.float32)
x_s = np.array([], dtype=np.float32)

def train(env_name, train=False, use_dumb=False, print_step_results=False, graph_target_fs=False):
    env = HvacEnv(use_dumb=use_dumb, print_step_results=print_step_results)

    global thermostat_temps
    thermostat_temps = np.empty((1, len(env.thermostats)), dtype='float32')
    
    states = len(env.thermostats)

    possible_actions = env.action_space

    agent = Agent(possible_actions, states, env_name, train)
        
    for episode in range(N_EPISODES):
        old_state = env.reset()

        total_reward = 0
        agent.n_episodes += 1

        is_done = False
    
        while not is_done:
            # uncomment below to view env
            # env.render()

            # log steps
            global times
            times = np.append(times, [env.step_count])

            actions = agent.get_actions(old_state, train=train)

            new_state, reward, done, _ = env.step(actions)

            if episode == (N_EPISODES / 2):
                env.scheduler.reset_scores()

            env.set_target_temp()
            env.set_outside_temp()

            is_done = done
            
            if train:
                agent.remember(old_state, actions, reward, new_state, done)

            # log thermostat temps
            env_thermostat_temps = env.get_thermostat_temps()
            thermostat_temps = np.append(
                thermostat_temps, 
                np.reshape(env_thermostat_temps, (1, len(env_thermostat_temps))), 
                axis=0
            )
            
            # log target temps
            global target_temps
            target_temps = np.append(target_temps, env.target_temp)

            # log rewards
            global rewards
            rewards = np.append(rewards, [reward])

            total_reward += reward

            old_state = new_state

        global episode_endpoints
        episode_endpoints = np.append(episode_endpoints, [times[len(times) - 1]])

        if train:
            agent.train_long_memory()
            agent.decrease_epsilon()
        
        if graph_target_fs:
            agent.trainer.update_target_fs()

        print("episode: {}/{} | score: {} | e: {:.3f}".format(episode +
              1, N_EPISODES, total_reward, agent.epsilon))
    
    global difference_sums
    difference_sums = env.scheduler.difference_sums

    global target_fs
    target_fs = agent.trainer.target_fs

    global x_s
    x_s = agent.trainer.x_s

    print('Difference Sums:\n{}'.format(env.scheduler.difference_sums))
    print('Total Difference Sums Score:\n{}'.format(env.scheduler.scores))
    print(env.success_count)

    if train:
        agent.model.save_weights("weights/" + env_name + ".h5", overwrite=True)

def to_csv():
    f = open('./graph.csv', 'w')
    writer = csv.writer(f)

    writer.writerow(rewards)
    writer.writerow(times)
    writer.writerow(episode_endpoints)

    for i in range(len(thermostat_temps[0])):
        sub_thermostat_temps = thermostat_temps[:, i][1:]
        writer.writerow(sub_thermostat_temps)

    for i in range(len(difference_sums[0])):
        sub_difference_sum = difference_sums[:, i][1:]
        writer.writerow(sub_difference_sum)
    
    f.close()

if __name__ == "__main__":
    graph_target_fs = False
    train('hvac', train=False, use_dumb=False, print_step_results=True, graph_target_fs=graph_target_fs)

    if not graph_target_fs:
        # graph
        # plt.plot(times, rewards, 'r', label='Reward')

        for i in range(len(thermostat_temps[0])):
            sub_thermostat_temps = thermostat_temps[:, i][1:]

            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)

            plt.plot(times, sub_thermostat_temps, c=color, label='Themostat: {}'.format(i))
        
        for i in range(len(difference_sums[0])):
            sub_difference_sum = thermostat_temps[:, i][1:]

            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)

            plt.plot(times, sub_difference_sum, c=color, label='Difference Sum: {}'.format(i))

        plt.plot(times, target_temps, c='b', label='Temperature Setpoint')

        plt.title('Reward & Thermostat')

        # for endpoint in episode_endpoints:
        #     plt.axvline(x=endpoint)
    else:
        for i in range(len(target_fs[0])):
            sub_target_fs = target_fs[:, i][1:]
            
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)

            plt.plot(x_s, sub_target_fs, c=color, label='Target_F: {}'.format(i))
        
        plt.title('Target_fs')

    plt.legend()

    to_csv()

    plt.show()

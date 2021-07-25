import sys
import random

import numpy as np

import csv
from matplotlib import pyplot as plt

from agent.agent import Agent

from env.hvac_env import HvacEnv

N_EPISODES = 10
MAX_TIMESTEP = 1000

episode_endpoints = np.array([], dtype='float32')
rewards = np.array([], dtype='float32')
times = np.array([], dtype='int32')
target_temps = np.array([], dtype='float32')

def train(env_name, train=True):
    env = HvacEnv()

    global target_temps
    target_temps = np.zeros((1, len(env.targets)), dtype='float32')
    
    states = env.observation_space.shape[0]
    p_actions = env.action_space

    agent = Agent(p_actions, states, env_name, train, ac_count=env.ac_count)

    index = 0
        
    for episode in range(N_EPISODES):
        old_state = env.reset()

        total_reward = 0
        agent.n_episodes += 1

        is_done = False
    
        while not is_done:
            # uncomment below to view env
            # env.render()

            index += 1

            global times
            times = np.append(times, [index])

            actions = agent.get_actions(old_state, train=train)

            new_state, reward, done, info = env.step(actions)
            agent.train_short_memory(old_state, actions, reward, new_state, done)

            is_done = done
            
            if train:
                agent.remember(old_state, actions, reward, new_state, done)

            target_temps = np.vstack((target_temps, env.target_temps()))

            global rewards
            rewards = np.append(rewards, [reward])
            total_reward += reward

            if reward < 0:
                print(reward)
                sys.exit()

            old_state = new_state

        global episode_endpoints
        episode_endpoints = np.append(episode_endpoints, [times[len(times) - 1]])

        if train:
            agent.train_long_memory()
        
        env.set_target_temp()

        print("episode: {}/{} | score: {} | e: {:.3f}".format(episode +
              1, N_EPISODES, total_reward, agent.epsilon))
        
    if train:
        agent.model.save_weights("weights/" + env_name + ".h5", overwrite=True)

def to_csv():
    f = open('./graph.csv', 'w')
    writer = csv.writer(f)

    writer.writerow(rewards)
    writer.writerow(target_temps)
    writer.writerow(times)
    writer.writerow(episode_endpoints)

    f.close()

if __name__ == "__main__":
    train('hvac', train=True)
    
    # graph
    plt.plot(times, rewards, 'r', label='Reward')
    
    for i in range(len(target_temps[0])):
        sub_target_temps = target_temps[:, i][1:]

        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)

        plt.plot(times, sub_target_temps, c=color, label='Themostat: {}'.format(i))
    
    plt.title('Reward & Thermostat')
    
    for endpoint in episode_endpoints:
        plt.axvline(x=endpoint)

    plt.legend()

    to_csv()

    plt.show()

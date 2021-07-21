import sys
import gym
import random

import numpy as np
from matplotlib import pyplot as plt

from agent.agent import Agent
from env.hvac_env import HvacEnv

N_EPISODES = 50
MAX_TIMESTEP = 1000

rewards = np.array([], dtype='float32')
times = np.array([], dtype='int32')
target_temps = np.array([], dtype='float32')

def train(env_name, train=True):
    env = HvacEnv()

    global target_temps
    target_temps = np.zeros((1, len(env.targets)), dtype='float32')
    
    states = env.observation_space.shape[1]
    p_actions = env.action_space

    agent = Agent(p_actions, states, env_name, train, ac_count=env.ac_count)

    index = 0

    old_state = env.reset().reshape(1, states)

    for episode in range(N_EPISODES):
        # uncomment below to view env
        # env.render()
        env.set_target_temp()
        print(env.target_temp)

        total_reward = 0
        agent.n_episodes += 1

        for t in range(MAX_TIMESTEP):
            index += 1

            global times
            times = np.append(times, [index])

            actions = agent.get_actions(old_state, train=train)

            new_state, reward, done, info = env.step(actions)
            new_state = new_state.reshape(1, states)
            
            if train:
                agent.remember(old_state, actions, reward, new_state, done)
            
            target_temps = np.vstack((target_temps, env.target_temps()))

            global rewards
            rewards = np.append(rewards, [reward])
            total_reward += reward

            old_state = new_state

            if done:
                break

        if train:
            agent.train_long_memory()

        print("episode: {}/{} | score: {} | e: {:.3f}".format(episode +
              1, N_EPISODES, total_reward, agent.epsilon))
    
    if train:
        agent.model.save_weights("weights/" + env_name + ".h5", overwrite=True)

if __name__ == "__main__":
    train('hvac', train=True)
    plt.plot(times, rewards, 'r', label='Reward')
    
    for i in range(len(target_temps[0])):
        sub_target_temps = target_temps[:, i][1:]

        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)

        plt.plot(times, sub_target_temps, c=color, label='Themostat: {}'.format(i))
    
    plt.title('Reward & Thermostat')

    plt.legend()
    plt.show()

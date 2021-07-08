from env.hvac_env import HvacEnv
import gym

from agent.agent import Agent

N_EPISODES = 1000
MAX_TIMESTEP = 1000

def train(env_name, train=True):
    env = HvacEnv()
    states = env.observation_space.shape[0]
    p_actions = env.action_space.n

    agent = Agent(p_actions, states, env_name, train)

    for episode in range(N_EPISODES):
        # uncomment below to view env
        # env.render() 
        old_state = env.reset().reshape(1, states)

        total_reward = 0
        agent.n_episodes += 1

        for t in range(MAX_TIMESTEP):
            actions = agent.get_actions(old_state, train=train)

            new_state, reward, done, info = env.step(actions)
            new_state = new_state.reshape(1, states)

            if train:
                agent.remember(old_state, actions, reward, new_state, done)

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

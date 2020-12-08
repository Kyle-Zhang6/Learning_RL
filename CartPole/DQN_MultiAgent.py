import gym
import numpy as np
from matplotlib import pyplot as plt
import RL_Functions


def extra_reward(observation, reward, episode_reward, done):
    if done and episode_reward < 500:
        r = -10
    else:
        r = 0
    return r


if __name__ == "__main__":
    ENV_NAME        = 'CartPole-v1'
    AGENT_TYPE      = 'DQN'
    AGENT_NUM       = 2
    LR              = 0.01
    TOTAL_EPISODES  = 300
    RENDER_GAP      = 20

    env = gym.make(ENV_NAME)
    model_kwargs = {'hidden_features': [64]}
    agents = [RL_Functions.agents.DQN_Agent(env, model_kwargs=model_kwargs, model=RL_Functions.MLP,
                                 agent_type=AGENT_TYPE, lr=0.01, epsilon=0.01) for _ in range(AGENT_NUM)]
    agents[0].epsilon = 0.
    replayer = RL_Functions.agents.Replayer(50000)
    for agent in agents:
        agent.replayer = replayer

    episode_rewards = []
    for i in range(TOTAL_EPISODES):
        if i % RENDER_GAP == (RENDER_GAP - 1):
            render = True
        else:
            render = False
        rewards = [RL_Functions.agents.play(env, agent, render=render, train=True, extra_reward_func=extra_reward)
                   for agent in agents]

        episode_rewards.append(rewards)
        print('No.{}: reward = {}'.format(i, rewards))
    episode_rewards = np.array(episode_rewards)
    print('\n\nAvg_reward = {}'.format(np.mean(episode_rewards, axis=0)))

    env.close()

    plt.plot(episode_rewards)
    plt.xlabel('Epoch_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('results/cartpool_v1_training_process_{}_MultiAgents_lrDescending.png'.format(AGENT_TYPE))
    plt.show()

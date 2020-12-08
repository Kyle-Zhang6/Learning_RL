import gym
import numpy as np
from matplotlib import pyplot as plt
import RL_Functions
from torch import optim, nn


def extra_reward(observation, reward, episode_reward, done):
    if done and episode_reward < 500:
        r = -10
    else:
        r = 0
    return r


if __name__ == "__main__":
    ENV_NAME = 'CartPole-v1'
    AGENT_TYPE = 'VPG'
    TOTAL_EPISODES = 3000
    RENDER_GAP = 300

    env = gym.make(ENV_NAME)
    model_kwargs = {'model': RL_Functions.MLP,
                    'hidden_features': [32],
                    'softmax': True,
                    'lr': 0.01,
                    'optimizer': optim.Adam}
    baseline_model_kwargs = {'model': RL_Functions.MLP,
                             'hidden_features': [32],
                             'lr': 0.01,
                             'optimizer': optim.Adam,
                             'criterion': nn.MSELoss()}

    agent = RL_Functions.agents.VPG_Agent(env, model_kwargs=model_kwargs,
                                          baseline_model_kwargs=baseline_model_kwargs)

    episode_rewards = []
    for i in range(TOTAL_EPISODES):
        if i % RENDER_GAP == (RENDER_GAP - 1):
            render = True
        else:
            render = False
        reward = RL_Functions.agents.play(env, agent, render=render, train=True, extra_reward_func=extra_reward)
        episode_rewards.append(reward)
        print('No.{}: reward = {}'.format(i, reward))

    print('\n\nAvg_reward = {}'.format(np.mean(episode_rewards)))
    env.close()

    if hasattr(agent, 'baseline_net'):
        AGENT_TYPE += '_baseline'
    plt.plot(episode_rewards)
    plt.xlabel('Epoch_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('results/cartpool_v1_training_process_{}.png'.format(AGENT_TYPE))
    plt.show()

    # Save Nets
    agent.save_net()

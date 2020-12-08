import gym
import numpy as np
from matplotlib import pyplot as plt
import RL_Functions


# To modify the raw reward to help train
def extra_reward(observation, reward, episode_reward, done):
    if done and episode_reward < 500:
        r = -10
    else:
        r = 0
    return r


if __name__ == "__main__":
    ENV_NAME        = 'CartPole-v1'
    AGENT_TYPE = 'DQN'
    LR              = 0.01
    TOTAL_EPISODES  = 300
    RENDER_GAP      = 20

    env = gym.make(ENV_NAME)
    model_kwargs = {'hidden_features': [64]}
    agent = RL_Functions.agents.DQN_Agent(env, model_kwargs=model_kwargs, model=RL_Functions.MLP,
                               agent_type=AGENT_TYPE, batch_size=64, lr=LR)

    episode_rewards = []
    a = 0
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

    plt.plot(episode_rewards)
    plt.xlabel('Epoch_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('results/cartpool_v1_training_process_{}.png'.format(AGENT_TYPE))
    plt.show()

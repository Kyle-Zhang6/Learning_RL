import gym
import RL_Functions
from torch import optim, nn
from matplotlib import pyplot as plt


def extra_reward(observation, reward, episode_reward, done):
    if done and episode_reward < 500:
        r = -10
    else:
        r = 0
    return r


if __name__ == "__main__":
    ENV_NAME         = 'CartPole-v0'
    AGENT_TYPE       = 'VPG'
    TOTAL_EPISODES   = 5000
    RENDER_GAP       = 500
    PRINT_GAP        = 20

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

    # Choose an agent with baseline or not
    # agent = RL_Functions.agents.VPG_Agent(env, model_kwargs=model_kwargs, baseline_model_kwargs=baseline_model_kwargs)
    agent = RL_Functions.agents.VPG_Agent(env, model_kwargs=model_kwargs, baseline_model_kwargs=None)

    episode_rewards = []
    episode_reward = 0
    for ep in range(TOTAL_EPISODES):
        if ep % RENDER_GAP == (RENDER_GAP - 1):
            render = True
        else:
            render = False

        # Agent interacting with the environment
        reward = RL_Functions.agents.play(env, agent, render=render, train=True, extra_reward_func=extra_reward)
        episode_reward += reward

        # Print process
        if ep % PRINT_GAP == (PRINT_GAP - 1):
            episode_rewards.append(episode_reward / PRINT_GAP)
            episode_reward = 0
            print('Episode NO.{} ~ {}: avg_reward = {}'.format(ep + 1 - PRINT_GAP, ep, episode_rewards[-1]))

    env.close()

    # Generate the figure
    if hasattr(agent, 'baseline_net'):
        AGENT_TYPE += '_baseline'
    plt.plot(episode_rewards)
    plt.xlabel('Epoch_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('./results/Training_Process_{}.png'.format(AGENT_TYPE))
    plt.show()

    # Save Nets
    # agent.save_net()

import gym
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from matplotlib import pyplot as plt


class Model(nn.Module):
    # Definition of Neural Network (a MLP model)
    def __init__(self, input_feature, hidden_feature, output_feature, lr=0.01):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_feature, hidden_feature)
        self.fc2 = nn.Linear(hidden_feature, output_feature)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x_ = F.relu(self.fc1(x))
        x_ = self.fc2(x_)
        return x_


class Actor:
    def __init__(self, input_feature, hidden_feature, output_feature, lr=0.01):
        self.net = Model(input_feature, hidden_feature, output_feature, lr=lr)

    def evaluate_action_prob(self, state, requires_grad=False):
        if requires_grad:
            self.net.train()
            action_prob = F.softmax(self.net(state), dim=-1)
        else:
            with torch.no_grad():
                self.net.eval()
                action_prob = F.softmax(self.net(state), dim=-1)
        return action_prob

    def decide(self, state):
        action_prob = self.evaluate_action_prob(state, requires_grad=False)
        p = Categorical(action_prob)
        action = p.sample().item()
        return action

    def learn(self, td_error, state, action):
        p = self.evaluate_action_prob(state, requires_grad=True)
        p = p.gather(1, action)
        loss = - td_error * torch.log(p)
        loss = loss.mean()
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


class Critic:
    def __init__(self, input_feature, hidden_feature, output_feature=1, lr=0.01):
        self.net = Model(input_feature, hidden_feature, output_feature, lr=lr)

    def evaluate_v(self, state, requires_grad=False):
        if requires_grad:
            self.net.train()
            v = self.net(state)
        else:
            with torch.no_grad():
                self.net.eval()
                v = self.net(state)
        return v

    def learn(self, state, target_U):
        v = self.evaluate_v(state, requires_grad=True)
        loss = F.mse_loss(v, target_U)
        loss = loss.mean()
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


class Memory:
    def __init__(self):
        self.trajectory = []

    def push(self, step):
        self.trajectory.append(step)

    def clear(self):
        self.trajectory = []

    def make_batch(self):
        s1_list, a_list, r_list, s2_list, done_list = [], [], [], [], []
        for t in self.trajectory[::-1]:
            s1, a, r, s2, done = t
            s1_list.append(s1)
            a_list.append([a])
            r_list.append([r])
            s2_list.append(s2)
            done_list.append([done])

        s1_list, s2_list = torch.FloatTensor(s1_list), torch.FloatTensor(s2_list)
        r_list, done_list = torch.FloatTensor(r_list), torch.FloatTensor(done_list)
        a_list = torch.tensor(a_list, dtype=torch.int64)
        self.clear()

        return s1_list, a_list, r_list, s2_list, done_list


def revise_reward(reward, steps, done):
    if done and steps < 200:
        reward -= 100
    return reward


if __name__ == '__main__':
    ENV_NAME        = 'CartPole-v0'
    TOTAL_EPISODE   = 10000
    RENDER_GAP      = 100
    GAMMA           = 0.99
    ACTOR_LR        = 0.0003
    CRITIC_LR       = 0.001
    TRAJECTORY_L    = 10  # Length of trajectory for each training step

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, 128, action_dim, lr=ACTOR_LR)
    critic = Critic(state_dim, 128, output_feature=1, lr=CRITIC_LR)  # Only output 1 score for the state
    memory = Memory()

    episode_rewards = []
    for ep in range(TOTAL_EPISODE):
        render = True if ((ep % RENDER_GAP) == RENDER_GAP - 1) else False
        done = False
        episode_reward = 0

        s1 = env.reset()
        while not done:
            for i in range(TRAJECTORY_L):
                if render:
                    env.render()
                # Get all useful information
                a = actor.decide(torch.FloatTensor(s1))
                s2, r, done, _ = env.step(a)
                episode_reward += r
                r = revise_reward(r, episode_reward, done)

                memory.push((s1, a, r, s2, done))
                s1 = s2
                if done:
                    break

            state, action, reward, state_next, done_ = memory.make_batch()
            # Train both the actor and the critic
            U = reward + GAMMA * critic.evaluate_v(state_next, requires_grad=False) * (1. - done_)
            v = critic.evaluate_v(state, requires_grad=False)
            td_error = U - v
            p = actor.evaluate_action_prob(state,requires_grad=False)
            actor.learn(td_error.detach(), state, action)
            critic.learn(state, U.detach())

        episode_rewards.append(episode_reward)
        print('Episode NO.{}: reward = {}'.format(ep+1, episode_reward))

    print('\nAverage Reward = {}'.format(np.mean(episode_rewards)))
    # Show the training process and save the fig
    plt.plot(episode_rewards)
    plt.xlabel('Episode_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('results/Training_Process_A2C.png')
    plt.show()

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
        self.fc3 = nn.Linear(hidden_feature, hidden_feature)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x_ = F.relu(self.fc1(x))
        # x_ = F.relu(self.fc3(x_))
        x_ = self.fc2(x_)
        return x_


class Actor:
    def __init__(self, input_feature, hidden_feature, output_feature, lr=0.01):
        self.net = Model(input_feature, hidden_feature, output_feature, lr=lr)
        self.net.to(DEVICE)

    def evaluate_action_prob(self, state, requires_grad=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if requires_grad:
            self.net.train()
            action_prob = F.softmax(self.net(state), dim=-1).squeeze(0)
        else:
            with torch.no_grad():
                self.net.eval()
                action_prob = F.softmax(self.net(state), dim=-1).squeeze(0)
        return action_prob

    def decide(self, state):
        action_prob = self.evaluate_action_prob(state, requires_grad=False)
        p = Categorical(action_prob)
        action = p.sample().item()
        return action

    def learn(self, discount, td_error, state, action):
        p = self.evaluate_action_prob(state, requires_grad=True)[action]
        loss = - discount * td_error * torch.log(p)
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


class Critic:
    def __init__(self, input_feature, hidden_feature, output_feature=1, lr=0.01):
        self.net = Model(input_feature, hidden_feature, output_feature, lr=lr)
        self.net.to(DEVICE)

    def evaluate_v(self, state, requires_grad=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if requires_grad:
            self.net.train()
            v = self.net(state).squeeze(0)
        else:
            with torch.no_grad():
                self.net.eval()
                v = self.net(state).squeeze(0)
        return v

    def learn(self, state, target_U):
        v = self.evaluate_v(state, requires_grad=True)
        loss = F.smooth_l1_loss(v, target_U)
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


def revise_reward(reward, steps, done):
    if done and steps < 200:
        reward -= 100
    return reward


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    ENV_NAME        = 'CartPole-v0'
    TOTAL_EPISODE   = 2000000
    RENDER_GAP      = 1000
    GAMMA           = 0.95
    ACTOR_LR        = 0.00005
    CRITIC_LR       = 0.0001

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, 128, action_dim, lr=ACTOR_LR)
    critic = Critic(state_dim, 128, output_feature=1, lr=CRITIC_LR)  # Only output 1 score for the state

    episode_rewards = []
    for ep in range(TOTAL_EPISODE):
        render = True if ((ep % RENDER_GAP) == RENDER_GAP - 1) else False
        done = False
        episode_reward = 0
        discount = 1.

        state = env.reset()
        while not done:
            if render:
                env.render()
            # Get all useful information
            action = actor.decide(state)
            state_next, reward, done, _ = env.step(action)
            episode_reward += reward
            reward = revise_reward(reward, episode_reward, done)
            # Train both the actor and the critic
            U = reward + GAMMA * critic.evaluate_v(state_next, requires_grad=False) * (1. - float(done))
            v = critic.evaluate_v(state, requires_grad=False)
            td_error = U - v
            p = actor.evaluate_action_prob(state,requires_grad=False)
            actor.learn(discount, td_error.detach(), state, action)
            critic.learn(state, U.detach())
            # Update
            discount *= GAMMA

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

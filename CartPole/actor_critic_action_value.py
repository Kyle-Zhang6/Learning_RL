import gym
import torch
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

    def learn(self, discount, q, state, action):
        p = self.evaluate_action_prob(state, requires_grad=True)[action]
        loss = - discount * q * torch.log(p)
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


class Critic:
    def __init__(self, input_feature, hidden_feature, output_feature, lr=0.01):
        self.net = Model(input_feature, hidden_feature, output_feature, lr=lr)
        self.net.to(DEVICE)

    def evaluate_q(self, state, action, requires_grad=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if requires_grad:
            self.net.train()
            qs = self.net(state).squeeze(0)
        else:
            with torch.no_grad():
                self.net.eval()
                qs = self.net(state).squeeze(0)
        return qs[action]

    def learn(self, state, action, target_U):
        q = self.evaluate_q(state, action, requires_grad=True)
        loss = F.mse_loss(q, target_U)
        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()


def revise_reward(reward, steps, done):
    if done and steps < 200:
        reward -= 10
    return reward


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    ENV_NAME        = 'CartPole-v0'
    TOTAL_EPISODE   = 5000
    RENDER_GAP      = 100
    PRINT_GAP       = 20
    GAMMA           = 0.95
    ACTOR_LR        = 0.0003
    CRITIC_LR       = 0.001

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, 64, action_dim, lr=ACTOR_LR)
    critic = Critic(state_dim, 64, action_dim, lr=CRITIC_LR)

    episode_rewards = []
    episode_reward = 0
    for ep in range(TOTAL_EPISODE):
        render = True if ((ep % RENDER_GAP) == RENDER_GAP - 1) else False
        done = False
        discount = 1.

        state = env.reset()
        action = actor.decide(state)
        while not done:
            if render:
                env.render()
            # Get all useful information
            state_next, reward, done, _ = env.step(action)
            action_next = actor.decide(state_next)
            episode_reward += reward
            reward = revise_reward(reward, episode_reward, done)
            # Train both the actor and the critic
            U = reward + GAMMA * critic.evaluate_q(state_next, action_next, requires_grad=False)
            q = critic.evaluate_q(state, action, requires_grad=True)
            actor.learn(discount, q.detach(), state, action)
            critic.learn(state, action, U.detach())
            p = actor.evaluate_action_prob(state, requires_grad=False)
            # Update
            discount *= GAMMA
            state, action = state_next, action_next

        if ep % PRINT_GAP == (PRINT_GAP - 1):
            episode_rewards.append(episode_reward / PRINT_GAP)
            episode_reward = 0
            print('Episode NO.{} ~ {}: avg_reward = {}'.format(ep + 1 - PRINT_GAP, ep, episode_rewards[-1]))

    # Show the training process and save the fig
    plt.plot(episode_rewards)
    plt.xlabel('Episode_Ind')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig('results/Training_Process_AC_action_value.png')
    plt.show()

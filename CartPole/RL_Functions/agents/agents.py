import pandas as pd
import numpy as np
import torch
import warnings
from RL_Functions.neuralnetwork import *


class Replayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.array(args, dtype=np.object)
        self.i += 1
        if self.i >= self.capacity:
            self.i = 0
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return tuple(np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQN_Agent:
    def __init__(self, env, model_kwargs, model=MLP, lr=0.01, gamma=0.99, epsilon=0.001, replayer_capacity=20000,
                 batch_size=64, agent_type='DQN'):
        self.agent_type = agent_type

        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = torch.tensor(gamma, dtype=torch.float)
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = Replayer(replayer_capacity)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_kwargs['input_feature'] = observation_dim
        model_kwargs['output_feature'] = self.action_n
        self.evaluate_net = model_kwargs['model'](model_kwargs)
        self.target_net = model_kwargs['model'](model_kwargs)
        self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.target_net.requires_grad_(False)
        self.evaluate_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def decide(self, observation, return_tensor=False):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float)
        ob = observation.view(1, -1).clone().detach()
        with torch.no_grad():
            self.evaluate_net.eval()
            action = torch.argmax(self.evaluate_net(ob))

        if return_tensor:
            return action
        else:
            return action.item()

    def learn(self, observation, action, reward, next_observation, done):
        ind = np.linspace(0, self.batch_size - 1, self.batch_size, dtype=np.int)
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        observations, rewards, next_observations = (torch.tensor(arg, dtype=torch.float).to(self.device)
                                                    for arg in
                                                    [observations, rewards, next_observations])
        next_actions_qs = self.target_net(next_observations)
        if self.agent_type == 'Double_DQN':
            target_actions = torch.tensor([self.decide(o) for o in observations]).to(self.device)
            next_max_qs = next_actions_qs[ind, target_actions]
        elif self.agent_type == 'DQN':
            next_max_qs = torch.max(next_actions_qs, dim=-1).values
        else:
            next_max_qs = torch.max(next_actions_qs, dim=-1).values
        us = rewards + self.gamma * next_max_qs
        us.requires_grad_(False)

        self.evaluate_net.train()
        self.optimizer.zero_grad()
        actions_qs = self.evaluate_net(observations)
        qs = actions_qs[ind, actions]
        loss = self.criterion(qs, us)
        loss.backward()
        self.optimizer.step()

        if done:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
            self.target_net.requires_grad_(False)
            self.target_net.eval()

    def save_net(self, path):
        self.evaluate_net.cpu()
        torch.save(self.evaluate_net.state_dict(), path)
        self.evaluate_net.to(self.device)


class VPG_Agent:
    def __init__(self, env, model_kwargs, baseline_model_kwargs=None, gamma=0.99):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.trace = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_kwargs['input_feature'] = observation_dim
        model_kwargs['output_feature'] = self.action_n
        self.net = model_kwargs['model'](model_kwargs)
        self.net.to(self.device)

        if baseline_model_kwargs is not None:
            if not ('model' in baseline_model_kwargs.keys()):
                baseline_model_kwargs['model'] = MLP
            baseline_model_kwargs['input_feature'] = observation_dim
            baseline_model_kwargs['output_feature'] = 1
            self.baseline_net = baseline_model_kwargs['model'](baseline_model_kwargs)
            self.baseline_net.to(self.device)

    def decide(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        with torch.no_grad():
            self.net.eval()
            actions_prob = self.net(state).numpy()
            action = np.random.choice(self.action_n, p=actions_prob.reshape(-1))
        return action

    def learn(self, state, action, reward, next_state, done):
        self.trace.append((state, action, reward))

        if done:
            df = pd.DataFrame(self.trace, columns=['states', 'actions', 'rewards'])
            df['discounts'] = self.gamma ** df.index.array
            df['discounted_rewards'] = df['discounts'] * df['rewards']
            df['Gs'] = df['discounted_rewards'][::-1].cumsum()  # [R0+rR1+r*rR2, rR1+r*rR2, r*rR2]
            df['Gs'] = df['Gs'] / df['discounts']  # [R0+rR1+r*rR2, R1+rR2, R2]
            states_tensor = torch.tensor(np.stack(df['states']), dtype=torch.float, requires_grad=False).to(self.device)
            discounts_tensor = torch.tensor(df['discounts'], dtype=torch.float, requires_grad=False).to(self.device)
            Gs_tensor = torch.tensor(df['Gs'], dtype=torch.float, requires_grad=False).to(self.device)

            if hasattr(self, 'baseline_net'):
                self.baseline_net.train()
                vs = self.baseline_net(states_tensor)
                loss = self.baseline_net.criterion(vs, Gs_tensor.view(-1, 1))
                self.baseline_net.optimizer.zero_grad()
                loss.backward()
                self.baseline_net.optimizer.step()
                Gs_tensor -= vs.view(-1)

            self.net.train()
            actions_probs = self.net(states_tensor)
            action_probs = torch.eye(self.action_n, dtype=torch.float).to(self.device)[df['actions']] * actions_probs
            action_probs = action_probs.sum(dim=-1)

            loss = -discounts_tensor * Gs_tensor.detach() * torch.log(action_probs)
            loss = loss.sum()
            self.net.optimizer.zero_grad()
            loss.backward()
            self.net.optimizer.step()

            self.trace = []

    def save_net(self, path='./VPG_Agent_net.pkl', save_baseline_net=True):
        self.net.cpu()
        torch.save(self.net.state_dict(), path)
        self.net.to(self.device)
        if save_baseline_net:
            if hasattr(self, 'baseline_net'):
                self.baseline_net.cpu()
                torch.save(self.baseline_net.state_dict(), path.replace('net', 'baseline_net'))
            else:
                warnings.warn('Net Saving Failure: no baseline_net in this agent.', UserWarning)

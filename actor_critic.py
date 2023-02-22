import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        conv_shape = self.calc_conv_output(input_dims)

        self.gru = nn.GRUCell(conv_shape, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def calc_conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)

        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)

        return int(np.prod(dims.size()))

    def forward(self, state, hx):
        state = T.tensor(state, dtype=T.float).unsqueeze(0)

        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        conv = F.elu(self.conv4(conv))

        conv_state = conv.view(288)
        hx = hx.view(256)

        hx = self.gru(conv_state, (hx))

        pi = self.pi(hx).unsqueeze(0)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action[0], v, log_prob, hx

    def calculate_discounted_reward(self, done, rewards, values):
        values = T.cat(values).squeeze()

        if len(values.size()) == 1: 
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).reshape(values.size())

        return batch_return

    def calculate_generalized_advantage_estimate(self, rewards, values):
        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)

        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        return gae

    def calc_cost(self, new_state, hx, done, rewards, values, log_probs, intrinsic_reward):
        rewards += intrinsic_reward.detach().numpy()

        returns = self.calculate_discounted_reward(done, rewards, values)

        next_v = T.zeros(1, 1)[0] if done else self.forward(new_state, hx)[1]

        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        gae = self.calculate_generalized_advantage_estimate(rewards, values)

        actor_loss = -(log_probs * gae).sum()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss

        return total_loss

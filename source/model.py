# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-05 15:11:20
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-05 15:50:07

import torch
import torch.nn as nn

class Actor(nn.Module):
	"""docstring for Actor"""
	def __init__(self):
		super(Actor, self).__init__()
		self.drop_layer = nn.Dropout(p=0.5)
		self.linear1 = nn.Linear(input_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, action_dim)
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)
		self.relu = nn.Relu()

		def forward(self, state, tanh=False):
			action = self.relu(self.linear1(state))
			action = self.drop_layer(action)
			action = self.relu(self.linear2(action))
			action = self.drop_layer(action)
			action = self.linear3(action)
			if tanh:
				action = F.tanh(action)
			return action

class Critic(nn.Module):
	"""docstring for Critic"""
	def __init__(self):
		super(Critic, self).__init__()
		self.drop_layer = nn.Dropout(p=0.5)
		self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)
	
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = F.relu(self.linear1(x))
		x = self.drop_layer(x)
		x = F.relu(self.linear2(x))
		x = self.drop_layer(x)
		x = self.linear3(x)
		return x
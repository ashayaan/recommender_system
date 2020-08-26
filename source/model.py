# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-05 15:11:20
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-08-18 10:54:49

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.ae = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.ae(x)
    
    def rec_error(self, x):
        error = torch.sum((x - self.ae(x)) ** 2, 1)
        if x.size(1) != 1:
            return error.detach()
        return error.item()

class Actor(nn.Module):
	"""docstring for Actor"""
	def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-1):
		super(Actor, self).__init__()
		#Droupout layer with 50% dropout probability
		self.dropout = nn.Dropout(p=0.5)
		
		self.layer1 = nn.Linear(input_dim, hidden_size)
		self.layer2 = nn.Linear(hidden_size, hidden_size)
		self.layer3 = nn.Linear(hidden_size, action_dim)
		
		#initializing weights for the third layer
		self.layer3.weight.data.uniform_(-init_w, init_w)
		self.layer3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state):
		# state = self.state_rep(state)

		x = F.relu(self.layer1(state))
		x = self.dropout(x)
		x = F.relu(self.layer2(x))
		x = self.dropout(x)
		x = self.layer3(x)
		return x

class Critic(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-5):
		super(Critic, self).__init__()
		#Dropout layer with 50% prob
		self.dropout = nn.Dropout(p=0.5)
		
		self.layer1 = nn.Linear(input_dim + action_dim, hidden_size)
		self.layer2 = nn.Linear(hidden_size, hidden_size)
		self.layer3 = nn.Linear(hidden_size, 1)
		
		#initializing weights for the third layer
		self.layer3.weight.data.uniform_(-init_w, init_w) 
		self.layer3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = F.relu(self.layer1(x))
		x = self.dropout(x)
		x = F.relu(self.layer2(x))
		x = self.dropout(x)
		x = self.layer3(x)
		return x
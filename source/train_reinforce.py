# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-13 10:30:21
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-28 14:48:27

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

import pandas as pd
import numpy as np
from environment import FrameEnvironment
from params import params
from model import Actor,Critic
import optimizer as optimizer_file
from scipy.spatial import distance
from plot import Plotter, pairwise_distances_fig, pairwise_distances, smooth\
				,smooth_gauss

from helper import prepareDataset,createDataTensor

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = cuda
writer = SummaryWriter(log_dir='../temp')
debug = {}


import warnings
warnings.filterwarnings("ignore")

# ---
frame_size = 10
batch_size = 10
n_epochs   = 100
plot_every = 30
step       = 0
num_items    = 5000 # n items to recommend. Can be adjusted for your vram 


class DiscreteActor(nn.Module):
	def __init__(self, hidden_size, num_inputs, num_actions):
		super(DiscreteActor, self).__init__()

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, num_actions)
		
		self.saved_log_probs = []
		self.rewards = []

	def forward(self, inputs):
		x = inputs
		x = F.relu(self.linear1(x))
		action_scores = self.linear2(x)
		return F.softmax(action_scores)
	
	
	def select_action(self, state):
		probs = self.forward(state)
		m = Categorical(probs)
		action = m.sample()
		self.saved_log_probs.append(m.log_prob(action))
		return action, probs

class ChooseREINFORCE():
	
	def __init__(self, method=None):
		if method is None:
			method = ChooseREINFORCE.reinforce
		self.method = method
	
	@staticmethod
	def basic_reinforce(policy, returns, *args, **kwargs):
		policy_loss = []
		for log_prob, R in zip(policy.saved_log_probs, returns):
			policy_loss.append(-log_prob * R)
		policy_loss = torch.cat(policy_loss).sum()
		return policy_loss
	
	@staticmethod
	def reinforce_with_correction():
		raise NotImplemented

	def __call__(self, policy, optimizer, learn=True):
		R = 0
		
		returns = []
		for r in policy.rewards[::-1]:
			R = r + 0.99 * R
			returns.insert(0, R)
			
		returns = torch.tensor(returns)
		returns = (returns - returns.mean()) / (returns.std() + 0.0001)

		policy_loss = self.method(policy, returns)
		
		if learn:
			optimizer.zero_grad()
			policy_loss.backward()
			optimizer.step()
		
		del policy.rewards[:]
		del policy.saved_log_probs[:]

		return policy_loss


def temporal_difference(reward, done, gamma, target):
	return reward + (1.0 - done) * gamma * target


def value_update(batch, params, nets, optimizer,
				 device=torch.device('cpu'),
				 debug=None, writer=None,
				 learn=False, step=-1):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state, action, reward, next_state, done = get_base_batch(batch, device=device)

	with torch.no_grad():
		next_action = nets['target_policy_net'](next_state)
		target_value = nets['target_value_net'](next_state, next_action.detach())
		expected_value = temporal_difference(reward, done, params['gamma'], target_value)
		expected_value = torch.clamp(expected_value,
									 params['min_value'], params['max_value'])

	value = nets['value_net'](state, action)

	value_loss = torch.pow(value - expected_value.detach(), 2).mean()

	if learn:
		optimizer['value_optimizer'].zero_grad()
		value_loss.backward(retain_graph=True)
		optimizer['value_optimizer'].step()

	elif not learn:
		debug['next_action'] = next_action
		writer.add_figure('next_action',
						  utils.pairwise_distances_fig(next_action[:50]), step)
		writer.add_histogram('value', value, step)
		writer.add_histogram('target_value', target_value, step)
		writer.add_histogram('expected_value', expected_value, step)

	return value_loss

def prepare_dataset(df, key_to_id, frame_size, env, sort_users=False, **kwargs):
	
	global num_items
	
	value_counts = df['movieId'].value_counts() 
	print('counted!')
	
	# here n items to keep are adjusted
	num_items = 5000
	to_remove = df['movieId'].value_counts().sort_values()[:-num_items].index
	to_keep = df['movieId'].value_counts().sort_values()[-num_items:].index
	to_remove_indices = df[df['movieId'].isin(to_remove)].index
	num_removed = len(to_remove)
	
	df.drop(to_remove_indices, inplace=True)
	print('dropped!')
	
	print('before', env.embeddings.size(), len(env.data))
	for i in list(env.data.keys()):
		if i not in to_keep:
			del env.data[i]
		
	env.embeddings, env.key_to_id, env.id_to_key = createDataTensor(env.data)
	
	print('after', env.embeddings.size(), len(env.data))
	print('embeddings automatically updated')
	print('action space is reduced to {} - {} = {}'.format(num_items + num_removed, num_removed,
														   num_items))
	
	return prepareDataset(df=df, key_to_id=env.key_to_id, environment=env,
									  frame_size=frame_size, sort_users=sort_users)

def batch_contstate_discaction(batch, item_embeddings_tensor, frame_size, num_items):
	
	items_t, ratings_t, sizes_t, users_t = batch['items'], batch['ratings'], batch['sizes'], batch['users']
	items_emb = item_embeddings_tensor[items_t.long()]
	b_size = ratings_t.size(0)

	items = items_emb[:, :-1, :].view(b_size, -1)
	next_items = items_emb[:, 1:, :].view(b_size, -1)
	ratings = ratings_t[:, :-1]
	next_ratings = ratings_t[:, 1:]

	state = torch.cat([items, ratings], 1)
	next_state = torch.cat([next_items, next_ratings], 1)
	action = items_t[:, -1]
	reward = ratings_t[:, -1]

	done = torch.zeros(b_size)
	done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1
	
	one_hot_action = torch.zeros(action.size(0), num_items)
	one_hot_action.scatter_(1, action.view(-1,1), 1)

	batch = {'state': state, 'action': one_hot_action, 'reward': reward, 'next_state': next_state, 'done': done,
			 'meta': {'users': users_t, 'sizes': sizes_t}}
	return batch

def embed_batch(batch, item_embeddings_tensor,frame_size):
	return batch_contstate_discaction(batch, item_embeddings_tensor, frame_size=frame_size, num_items=num_items)

def softUpdate(net, target_net, soft_tau=1e-2):
	for target_param, param in zip(target_net.parameters(), net.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def get_base_batch(batch, device, done=True):
	b = [batch['state'], batch['action'], batch['reward'].unsqueeze(1), batch['next_state'], ]
	if done:
		b.append(batch['done'].unsqueeze(1))
	else:
		batch.append(torch.zeros_like(batch['reward']))
	return [i.to(device) for i in b]


def write_losses(writer, loss_dict, kind='train'):

	def write_loss(kind, key, item, step):
		writer.add_scalar(kind + '/' + key, item, global_step=step)

	step = loss_dict['step']
	for k, v in loss_dict.items():
		if k == 'step':
			continue
		write_loss(kind, k, v, step)

	writer.close()

def reinforce_update(batch, params, nets, optimizer,
					 device=torch.device('cpu'),
					 debug=None, writer=None,
					 learn=False, step=-1):
	torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state, action, reward, next_state, done = get_base_batch(batch,device=device)
	
	predicted_action, predicted_probs = nets['policy_net'].select_action(state)
	reward = nets['value_net'](state, predicted_probs).detach()
	print(reward.mean())
	nets['policy_net'].rewards.append(reward.mean())
	
	value_loss = value_update(batch, params, nets, optimizer,
					 writer=writer,
					 device=device,
					 debug=debug, learn=learn, step=step)
	
	
	
	if step % params['policy_step'] == 0 and step > 0:
		
		policy_loss = params['reinforce'](nets['policy_net'], optimizer['policy_optimizer'], learn=learn)
		
		del nets['policy_net'].rewards[:]
		del nets['policy_net'].saved_log_probs[:]
		
		print('step: ', step, '| value:', value_loss.item(), '| policy', policy_loss.item())
	
		softUpdate(nets['value_net'], nets['target_value_net'], soft_tau=params['soft_tau'])
		softUpdate(nets['policy_net'], nets['target_policy_net'], soft_tau=params['soft_tau'])

		losses = {'value': value_loss.item(),
				  'policy': policy_loss.item(),
				  'step': step}

		write_losses(writer, losses, kind='train' if learn else 'test')

		return losses

if __name__ == '__main__':
	env = FrameEnvironment('../data/ml20_pca128.pkl','../data/ml-20m/ratings.csv', frame_size, batch_size,embed_batch=embed_batch, prepare_dataset=prepare_dataset,num_workers = 0)
	params = {
	'reinforce': ChooseREINFORCE(ChooseREINFORCE.basic_reinforce),
	'gamma'      : 0.99,
	'min_value'  : -10,
	'max_value'  : 10,
	'policy_step': 10,
	'soft_tau'   : 0.001,
	
	'policy_lr'  : 1e-5,
	'value_lr'   : 1e-5,
	'actor_weight_init': 54e-2,
	'critic_weight_init': 6e-1,
	}

	nets = {
	'value_net': Critic(1290, num_items, 2048, params['critic_weight_init']).to(cuda),
	'target_value_net': Critic(1290, num_items, 2048, params['actor_weight_init']).to(cuda).eval(),
	
	'policy_net':  DiscreteActor(2048, 1290, num_items).to(cuda),
	'target_policy_net': DiscreteActor(2048, 1290, num_items).to(cuda).eval(),
	}


	# from good to bad: Ranger Radam Adam RMSprop
	optimizer = {
		'value_optimizer': optimizer_file.Ranger(nets['value_net'].parameters(),
											  lr=params['value_lr'], weight_decay=1e-2),

		'policy_optimizer': optimizer_file.Ranger(nets['policy_net'].parameters(),
											   lr=params['policy_lr'], weight_decay=1e-5)
	}


	loss = {
		'test': {'value': [], 'policy': [], 'step': []},
		'train': {'value': [], 'policy': [], 'step': []}
		}
	plotter = Plotter(loss, [['value', 'policy']],)
	step = 0
	for epoch in range(n_epochs):
		for batch in env.train_dataloader:
			loss = reinforce_update(batch, params, nets, optimizer,
						 writer=writer,
						 device=device,
						 debug=debug, learn=True, step=step)
			if loss:
				plotter.log_losses(loss)
			step += 1
			if step % plot_every == 0:
				print('step', step)
				# plotter.log_losses(test_loss, test=False)
				plotter.plot_loss()
			if step > 1000:
			   pass
			   assert False


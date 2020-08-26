# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-13 10:30:21
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-28 13:29:17

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from environment import FrameEnvironment
from params import params
from model import Actor,Critic
import optimizer
from scipy.spatial import distance
from plot import Plotter, pairwise_distances_fig, pairwise_distances, smooth\
				,smooth_gauss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global variables
writer = SummaryWriter(log_dir='../temp')
debug = {}



def softUpdate(net, target_net, soft_tau=1e-2):
	for target_param, param in zip(target_net.parameters(), net.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
			
def run_tests(env,step,value_net,policy_net,target_value_net,target_policy_net,\
					value_optimizer, policy_optimizer, plotter):
	
	test_batch = next(iter(env.test_dataloader))
	losses = ddpg(value_net,policy_net,target_value_net,target_policy_net,\
					value_optimizer, policy_optimizer,test_batch, params, learn=False, step=step)
	
	gen_actions = debug['next_action']
	true_actions = env.embeddings.detach().cpu().numpy()
	
	# f = plotter.kde_reconstruction_error(ad, gen_actions, true_actions, cuda)
	# writer.add_figure('rec_error',f, losses['step'])
	return losses


def get_base_batch(batch, device=device, done=True):
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

def ddpg(value_net, policy_net, target_value_net, target_policy_net, value_optimizer,\
				 policy_optimizer, batch, params, learn=True, step=-1):
	
	state, action, reward, next_state, done = get_base_batch(batch)

	with torch.no_grad():
		next_action = target_policy_net(next_state)
		target_value   = target_value_net(next_state, next_action.detach())
		#Using the mean squared Bellman equation to get expected reward
		expected_value = reward + (1.0 - done) * params['gamma'] * target_value
		expected_value = torch.clamp(expected_value,params['min_value'], params['max_value'])

	value = value_net(state, action)
	
	#How far the value is from the optimal policy
	value_loss = torch.pow(value - expected_value.detach(), 2).mean()
	
	if learn:
		value_optimizer.zero_grad()
		value_loss.backward()
		value_optimizer.step()
	else:
		debug['next_action'] = next_action
		writer.add_figure('next_action',pairwise_distances_fig(next_action[:50]), step)
		writer.add_histogram('value', value, step)
		writer.add_histogram('target_value', target_value, step)
		writer.add_histogram('expected_value', expected_value, step)
	
	# --------------------------------------------------------#
	# Policy learning
	
	gen_action = policy_net(state)
	policy_loss = -value_net(state, gen_action)
	
	if not learn:
		debug['gen_action'] = gen_action
		writer.add_histogram('policy_loss', policy_loss, step)
		writer.add_figure('next_action',
					pairwise_distances_fig(gen_action[:50]), step)
		
	policy_loss = policy_loss.mean()
	
	if learn and step % params['policy_step']== 0:
		policy_optimizer.zero_grad()
		policy_loss.backward()
		torch.nn.utils.clip_grad_norm_(policy_net.parameters(), -1, 1)
		policy_optimizer.step()

		softUpdate(value_net, target_value_net, soft_tau=params['soft_tau'])
		softUpdate(policy_net, target_policy_net, soft_tau=params['soft_tau'])

	losses = {'value': value_loss.item(), 'policy': policy_loss.item(), 'step': step}
		
	if learn and step % params['policy_step']== 0:
		write_losses(writer, losses, kind='train' if learn else 'test')
	
	if learn:
		return losses, value_net, policy_net, target_value_net, target_policy_net, value_optimizer, policy_optimizer
	else:
		return losses

def train(env):
	value_net  = Critic(1290, 128, 256, params['critic_weight_init']).to(device)
	policy_net = Actor(1290, 128, 256, params['actor_weight_init']).to(device)
	target_value_net = Critic(1290, 128, 256).to(device)
	target_policy_net = Actor(1290, 128, 256).to(device)

	#Switiching off dropout layers
	target_value_net.eval()
	target_policy_net.eval()

	softUpdate(value_net, target_value_net, soft_tau=1.0)
	softUpdate(policy_net, target_policy_net, soft_tau=1.0)

	value_optimizer = optimizer.Ranger(value_net.parameters(),lr=params['value_lr'], weight_decay=1e-2)
	policy_optimizer = optimizer.Ranger(policy_net.parameters(),lr=params['policy_lr'], weight_decay=1e-5)
	value_criterion = nn.MSELoss()
	loss = {'test': {'value': [], 'policy': [], 'step': []},'train': {'value': [], 'policy': [], 'step': []}}
	
	plotter = Plotter(loss, [['value', 'policy']],)

	step = 0
	plot_every = 10
	for epoch in range(100):
		print("Epoch: {}".format(epoch+1))
		for batch in (env.train_dataloader):
			loss, value_net, policy_net, target_value_net, target_policy_net, value_optimizer, policy_optimizer\
			 = ddpg(value_net,policy_net,target_value_net,target_policy_net,\
					value_optimizer, policy_optimizer, batch, params, step=step)
			# print(loss)
			plotter.log_losses(loss)
			step += 1
			if step % plot_every == 0:
				print('step', step)
				test_loss = run_tests(env,step,value_net,policy_net,target_value_net,target_policy_net,\
					value_optimizer, policy_optimizer,plotter)
				plotter.log_losses(test_loss, test=True)
				plotter.plot_loss()
			if step > 1500:
				assert False


if __name__ == '__main__':
	environment = FrameEnvironment('../data/ml20_pca128.pkl','../data/ml-20m/ratings.csv')
	train(environment)
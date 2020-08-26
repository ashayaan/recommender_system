# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-08-03 11:18:22
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-08-18 12:55:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import pickle
import gc
import json

from environment import FrameEnvironment
from model import Actor,Critic,AnomalyDetector
import optimizer
from plot import Plotter, pairwise_distances_fig, pairwise_distances, smooth\
				,smooth_gauss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frame_size = 10
batch_size = 25
n_epochs   = 1
plot_every = 30
step       = 0

params = {
	'gamma':0.99,
	'soft_tau':0.001,
	'n_generator_samples':3,
	'perturbator_step': 2,
	'perturbator_lr':1e-5,
	'value_lr':1e-5,
	'generator_lr':1e-3,
}

# writer = SummaryWriter(log_dir='../temp')

writer = SummaryWriter(log_dir='../temp')
debug = {}

def run_tests(env,params,writer,debug):
	test_batch = next(iter(env.test_dataloader))
	losses = bcq_update(test_batch, params, writer, debug, learn=False, step=step)
	
	gen_actions = debug['perturbed_actions']
	true_actions = env.embeddings.detach().cpu().numpy()
	
	f = plotter.kde_reconstruction_error(ad, gen_actions, true_actions, device)
	writer.add_figure('rec_error',f, losses['step'])
	
	return losses

def soft_update(net, target_net, soft_tau=1e-2):
	for target_param, param in zip(target_net.parameters(), net.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def write_losses(writer, loss_dict, kind='train'):

	def write_loss(kind, key, item, step):
		writer.add_scalar(kind + '/' + key, item, global_step=step)

	step = loss_dict['step']
	for k, v in loss_dict.items():
		if k == 'step':
			continue
		write_loss(kind, k, v, step)

	writer.close()

def get_base_batch(batch, device=device, done=True):
	b = [batch['state'], batch['action'], batch['reward'].unsqueeze(1), batch['next_state'], ]
	if done:
		b.append(batch['done'].unsqueeze(1))
	else:
		batch.append(torch.zeros_like(batch['reward']))
	return [i.to(device) for i in b]


class Generator(nn.Module):
	def __init__(self, input_dim, action_dim, latent_dim):
		super(Generator, self).__init__()
		#encoder
		self.e1 = nn.Linear(input_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)
		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)
		
		#decoder
		self.d1 = nn.Linear(input_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)
		
		self.latent_dim = latent_dim
		self.normal = torch.distributions.Normal(0, 1)


	def forward(self, state, action):
		# z is encoded state + action
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * self.normal.sample(std.size()).to(device)

		# u is decoded action
		u = self.decode(state, z)

		return u, mean, std
	
	def decode(self, state, z=None):
		if z is None:
			z = self.normal.sample([state.size(0), self.latent_dim])
			z = z.clamp(-0.5, 0.5).to(device)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.d3(a)

class Perturbator(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-1):
		super(Perturbator, self).__init__()
		
		self.drop_layer = nn.Dropout(p=0.5)
		
		self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, action_dim)
		
		self.linear3.weight.data.uniform_(-init_w, init_w)
		self.linear3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state, action):
		a = torch.cat([state, action], 1)
		a = F.relu(self.linear1(a))
		a = self.drop_layer(a)
		a = F.relu(self.linear2(a))
		a = self.drop_layer(a)
		a = self.linear3(a) 
		return a + action



def bcq_update(batch, params, writer, debug, learn=True, step=-1):
	
	state, action, reward, next_state, done = get_base_batch(batch)
	batch_size = done.size(0)
	
	recon, mean, std = generator_net(state, action)
	recon_loss = F.mse_loss(recon, action)
	KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
	generator_loss = recon_loss + 0.5 * KL_loss
	
	if not learn:
	# 	writer.add_histogram('generator_mean', mean,step)
	# 	writer.add_histogram('generator_std', std, step)
		debug['recon'] = recon
	# 	writer.add_figure('reconstructed',pairwise_distances_fig(recon[:50]), step)
		
	if learn:
		generator_optimizer.zero_grad()
		generator_loss.backward()
		generator_optimizer.step()
	# --------------------------------------------------------#
	# Value Learning
	with torch.no_grad():
		# p.s. repeat_interleave was added in torch 1.1
		# if an error pops up, run 'conda update pytorch'
		state_rep = torch.repeat_interleave(next_state, params['n_generator_samples'], 0)
		sampled_action = generator_net.decode(state_rep)
		perturbed_action = target_perturbator_net(state_rep, sampled_action)
		target_Q1 = target_value_net1(state_rep, perturbed_action)
		target_Q2 = target_value_net1(state_rep, perturbed_action)
		target_value = 0.75 * torch.min(target_Q1, target_Q2) # value soft update
		target_value+= 0.25 * torch.max(target_Q1, target_Q2) #
		target_value = target_value.view(batch_size, -1).max(1)[0].view(-1, 1)
		
		expected_value = reward + (1.0 - done) * params['gamma'] * target_value

	value = value_net1(state, action)
	value_loss = torch.pow(value - expected_value.detach(), 2).mean()
	
	if learn:
		value_optimizer1.zero_grad()
		value_optimizer2.zero_grad()
		value_loss.backward()
		value_optimizer1.step()
		value_optimizer2.step()
	# else:
	# 	writer.add_histogram('value', value, step)
	# 	writer.add_histogram('target_value', target_value, step)
	# 	writer.add_histogram('expected_value', expected_value, step)
	# 	writer.close()
	
	sampled_actions = generator_net.decode(state)
	perturbed_actions= perturbator_net(state, sampled_actions)
	perturbator_loss = -value_net1(state, perturbed_actions)
	# if not learn:
	# 	writer.add_histogram('perturbator_loss', perturbator_loss, step)
	perturbator_loss = perturbator_loss.mean()
	
	if learn:
		if step % params['perturbator_step']:
			perturbator_optimizer.zero_grad()
			perturbator_loss.backward()
			torch.nn.utils.clip_grad_norm_(perturbator_net.parameters(), -1, 1)
			perturbator_optimizer.step()
		
		soft_update(value_net1, target_value_net1, soft_tau=params['soft_tau'])
		soft_update(value_net2, target_value_net2, soft_tau=params['soft_tau'])
		soft_update(perturbator_net, target_perturbator_net, soft_tau=params['soft_tau'])
	else:
		debug['sampled_actions'] = sampled_actions
		debug['perturbed_actions'] = perturbed_actions
	# 	writer.add_figure('sampled_actions',pairwise_distances_fig(sampled_actions[:50]), step)
	# 	writer.add_figure('perturbed_actions',pairwise_distances_fig(perturbed_actions[:50]), step)

	losses = {'value': value_loss.item(),
			  'perturbator': perturbator_loss.item(),
			  'generator': generator_loss.item(),
			  'step': step}
	
	# write_losses(writer, losses, kind='train' if learn else 'test')
	
	return losses

if __name__ == '__main__':
	env = FrameEnvironment('../data/ml20_pca128.pkl','../data/ml-20m/ratings.csv',frame_size, batch_size)
	
	generator_net = Generator(1290, 128, 512).to(device)
	value_net1  = Critic(1290, 128, 256, init_w=8e-1).to(device)
	value_net2  = Critic(1290, 128, 256, init_w=8e-1).to(device)
	perturbator_net = Perturbator(1290, 128, 256, init_w=27e-2).to(device)

	target_value_net1 = Critic(1290, 128, 256).to(device)
	target_value_net2 = Critic(1290, 128, 256).to(device)
	target_perturbator_net = Perturbator(1290, 128, 256).to(device)

	ad = AnomalyDetector().to(device)
	ad.load_state_dict(torch.load('trained/anomaly.pt'))
	ad.eval()

	target_perturbator_net.eval()
	target_value_net1.eval()
	target_value_net2.eval()

	soft_update(value_net1, target_value_net1, soft_tau=1.0)
	soft_update(value_net2, target_value_net2, soft_tau=1.0)
	soft_update(perturbator_net, target_perturbator_net, soft_tau=1.0)

	# optim.Adam can be replaced with RAdam
	value_optimizer1 = optimizer.Ranger(value_net1.parameters(), lr=params['value_lr'], k=10)
	value_optimizer2 = optimizer.Ranger(value_net2.parameters(), lr=params['perturbator_lr'], k=10)
	perturbator_optimizer = optimizer.Ranger(perturbator_net.parameters(), lr=params['value_lr'], weight_decay=1e-3,k=10)
	generator_optimizer = optimizer.Ranger(generator_net.parameters(), lr=params['generator_lr'], k=10)
	
	loss = {
		'train': {'value': [], 'perturbator': [], 'generator': [], 'step': []},
		'test': {'value': [], 'perturbator': [], 'generator': [], 'step': []},
		}
	
	plotter = Plotter(loss, [['generator'], ['value', 'perturbator']])


	for epoch in range(n_epochs):
		print("Epoch: {}".format(epoch+1))
		for batch in env.train_dataloader:
			loss = bcq_update(batch, params, writer, debug, step=step)
			plotter.log_losses(loss)
			step += 1
			print("Loss:{}".format(loss))
			if step % plot_every == 0:
				print('step', step)
				test_loss = run_tests(env,params,writer,debug)
				print(test_loss)
				plotter.log_losses(test_loss, test=True)
				plotter.plot_loss()
			if step > 1500:
				break
	gen_actions = debug['perturbed_actions']
	true_actions = env.embeddings.numpy()


	ad = AnomalyDetector().to(device)
	ad.load_state_dict(torch.load('trained/anomaly.pt'))
	ad.eval()

	plotter.plot_kde_reconstruction_error(ad, gen_actions, true_actions, device)


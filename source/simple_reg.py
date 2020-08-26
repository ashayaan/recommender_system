# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-08-18 10:55:19
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-08-18 14:04:39


import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import pickle
import json

from model import AnomalyDetector
import optimizer

from scipy import stats



cuda = torch.device('cpu')
frame_size = 10




if __name__ == '__main__':
	movies = pickle.load(open('../data/ml20_pca128.pkl','rb'))
	
	for i in movies.keys():
		movies[i] = movies[i].to(cuda)

	data = torch.stack(list(movies.values())).to(cuda)
	data = data[torch.randperm(data.size()[0])] # shuffle rows
	data_test = data[-100:]
	data = data[:-100]
	n_epochs = 5000
	batch_size = 15000

	model = AnomalyDetector().to(cuda)
	criterion = nn.MSELoss()
	optimizer = optimizer.Ranger(model.parameters(), lr=1e-4, weight_decay=1e-2)
	run_loss = []

	test_loss = []
	rec_loss = []
	test_rec_loss = []

	for epoch in tqdm(range(n_epochs)):
		for batch in data.split(batch_size):
			optimizer.zero_grad()
			batch = batch
			output = model(batch).float()
			loss = criterion(output, batch)
			test_loss.append(criterion(model(data_test).float(), data_test).item())
			rec_loss.append(model.rec_error(batch))
			test_rec_loss.append(model.rec_error(data_test))
			loss.backward()
			optimizer.step()
			run_loss.append(loss.item())
			print('Epoch:{} loss:{}'.format(epoch,loss.item()))


	plt.plot(run_loss,label='train_loss')
	plt.plot(test_loss,label='test_loss')
	plt.xlabel('epoch')
	plt.ylabel('MSE')
	plt.legend()
	plt.show()

	def calc_art_score(x):
		return model.rec_error(x)  + (1 / x.var() * 5)

	model.eval()
	train_scores = model.rec_error(data).detach().cpu().numpy()
	print(train_scores)


	train_scores = model.rec_error(data).detach().cpu().numpy()
	train_kernel = stats.gaussian_kde(train_scores)
	test_scores = model.rec_error(data_test).detach().cpu().numpy()
	test_kernel = stats.gaussian_kde(test_scores)
	x = np.linspace(0,1000, 100)
	probs_train = train_kernel(x)
	probs_test = test_kernel(x)
	plt.plot(x, probs_train, '-b', label='train dist')
	plt.plot(x, probs_test, '-r', label='test dist')
	plt.legend()
	plt.show()


	torch.save(model.state_dict(), "trained/anomaly.pt")
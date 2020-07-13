# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-12 14:44:07
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-13 12:38:31


import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from helper import prepareDataset,createBatchTensor,createDataTensor,\
					rolling_window,sortUsersItemwise,prepareBatchDynamic,\
					padder, batchStatistics


#Custom data set function to return data
class UserDataset(Dataset):
	def __init__(self, users, user_dict):
		self.users = users
		self.user_dict = user_dict

	def __len__(self):
		
		return len(self.users)

	def __getitem__(self, idx):
		idx = self.users[idx]
		group = self.user_dict[idx]
		items = group['items'][:]
		rates = group['ratings'][:]
		size = items.shape[0]
		return {'items': items, 'rates': rates, 'sizes': size, 'users': idx}



class Environment(object):
	"""docstring for Environment"""
	def __init__(self, embeddings, ratings, test_size=0.05, min_seq_size=10,
				prepare_dataset=prepareDataset,embed_batch=createBatchTensor):
		super(Environment, self).__init__()
		
		self.prepare_dataset = prepare_dataset
		self.embed_batch = embed_batch
		self.data = pickle.load(open(embeddings,'rb'))
		self.embeddings,self.key_to_id,self.id_to_key = createDataTensor(self.data)
		
		self.ratings = pd.read_csv(ratings)
		self.user_dict = None
		self.users = None
		self.prepare_dataset(self.ratings,self.key_to_id,min_seq_size,self)


		self.train_users, self.test_users = train_test_split(self.users, test_size=test_size)
		self.train_users = sortUsersItemwise(self.user_dict, self.train_users)[2:]
		self.test_users = sortUsersItemwise(self.user_dict, self.test_users)

		self.train_user_dataset = UserDataset(self.train_users, self.user_dict)
		self.test_user_dataset = UserDataset(self.test_users, self.user_dict)


class FrameEnvironment(Environment):
	"""docstring for FrameEnvironment"""
	def __init__(self, embeddings, ratings, frame_size=10, batch_size=25, num_workers=1):
		super(FrameEnvironment, self).__init__(embeddings, ratings, min_seq_size=frame_size+1)
		
		def prepareBatchCalls(x):
			batch = batchStatistics(x, self.embeddings,embed_batch=self.embed_batch,frame_size=frame_size)
			return batch

		self.prepareBatchCalls = prepareBatchCalls
		self.frame_size = frame_size
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.train_dataloader = DataLoader(self.train_user_dataset, batch_size=batch_size,
										   shuffle=True, num_workers=num_workers, collate_fn=prepareBatchCalls)

		self.test_dataloader = DataLoader(self.test_user_dataset, batch_size=batch_size,
										  shuffle=True, num_workers=num_workers, collate_fn=prepareBatchCalls)


	def train_batch(self):
		return next(iter(self.train_dataloader))

	def test_batch(self):
		return next(iter(self.test_dataloader))


#Unit testing
if __name__ == '__main__':
	test = FrameEnvironment('../data/ml20_pca128.pkl','../data/ml-20m/ratings.csv')
	x = test.train_batch()
	print(x['next_state'].shape)


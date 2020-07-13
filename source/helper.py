# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-12 14:58:57
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-13 14:39:26

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm




def rolling_window(data, window):
	shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
	strides = data.strides + (data.strides[-1],)
	return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def sortUsersItemwise(user_dict, users):
	return pd.Series(dict([(i, user_dict[i]['items'].shape[0]) for i in users])).sort_values(ascending=False).index


#Creates a stack of tensors of the embeddings of 
#the features of the movies
def createDataTensor(data):
	keys = list(sorted(data.keys()))
	key_to_id = dict(zip(keys,range( len(keys) ) ))
	id_to_key = dict(zip(range( len(keys) ),keys) ) 

	embeddings_id_dict = {}

	for i in data.keys():
		embeddings_id_dict[key_to_id[i]] = data[i]

	embeddings_tensor = torch.stack([embeddings_id_dict[i] for i in range(len(embeddings_id_dict))])
	return embeddings_tensor, key_to_id, id_to_key



def createBatchTensor(batch, embeddings_tensor, frame_size):
	items_t, ratings_t, sizes_t, users_t = batch['items'], batch['ratings'], batch['sizes'], batch['users']
	items_emb = embeddings_tensor[items_t.long()]
	b_size = ratings_t.size(0)

	items = items_emb[:, :-1, :].view(b_size, -1)
	next_items = items_emb[:, 1:, :].view(b_size, -1)
	ratings = ratings_t[:, :-1]
	next_ratings = ratings_t[:, 1:]

	# print(next_ratings[1])
	# print(items.shape, next_items.shape, ratings.shape, next_ratings.shape)


	state = torch.cat([items, ratings], 1)
	next_state = torch.cat([next_items, next_ratings], 1)
	action = items_emb[:, -1, :]
	reward = ratings_t[:, -1]

	done = torch.zeros(b_size)
	done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

	batch = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done,
			 'meta': {'users': users_t, 'sizes': sizes_t}}
	# print('\n')
	return batch



def prepareDataset(df,key_to_id,frame_size,environment,sort_users=False):
	df['rating'] = df['rating'].apply(lambda x: 2*(x-2.5)) #changing the ratings to -5-5
	df['movieId'] = df['movieId'].apply(lambda x: key_to_id.get(x))

	users = df[['userId', 'movieId']].groupby(['userId']).size()
	users = users[users > frame_size] #We are only taking users who have rated some threshold

	if sort_users == True:
		users = users.sort_values(ascending=False)
	users = users.index
	ratings = df.sort_values(by='timestamp').set_index('userId').drop('timestamp', axis=1).groupby('userId')

	user_dict = {}
	
	def prepareDatasetHelper(x):
		userid = x.index[0]
		user_dict[int(userid)] = {}
		user_dict[int(userid)]['items'] = x['movieId'].values
		user_dict[int(userid)]['ratings'] = x['rating'].values

	ratings.apply(prepareDatasetHelper)

	environment.user_dict = user_dict
	environment.users = users

	return {'df': df, 'key_to_id': key_to_id,'frame_size': frame_size, 'env': environment, 'sort_users': sort_users}

def padder(x):
	items = []
	ratings = []
	sizes = []
	users = []
	for i in range(len(x)):
		items.append(torch.tensor(x[i]['items']))
		ratings.append(torch.tensor(x[i]['rates']))
		sizes.append(x[i]['sizes'])
		users.append(x[i]['users'])
	items = torch.nn.utils.rnn.pad_sequence(items, batch_first=True).long()
	ratings = torch.nn.utils.rnn.pad_sequence(ratings, batch_first=True).float()
	sizes = torch.tensor(sizes).float()
	return {'items': items, 'ratings': ratings, 'sizes': sizes, 'users': users}

def prepareBatchDynamic(batch, embeddings_tensor, embed_batch=None):
	item_idx, ratings, sizes, users = batch['items'], batch['ratings'], batch['sizes'], batch['users']
	item = embeddings_tensor[item_idx]
	batch = {'items': item, 'users': users, 'ratings': ratings, 'sizes': sizes}
	return batch

def batchStatistics(batch,embeddings_tensor,frame_size,embed_batch):
	item, ratings, sizes, users = [], [], [], []
	for i in range(len(batch)):
		item.append(batch[i]['items'])
		ratings.append(batch[i]['rates'])
		sizes.append(batch[i]['sizes'])
		users.append(batch[i]['users'])

	item = np.concatenate([rolling_window(i, frame_size + 1) for i in item], 0)
	ratings = np.concatenate([rolling_window(i, frame_size + 1) for i in ratings], 0)

	item = torch.tensor(item)
	users = torch.tensor(users)
	ratings = torch.tensor(ratings).float()
	sizes = torch.tensor(sizes)

	batch = {'items': item, 'users': users, 'ratings': ratings, 'sizes': sizes}

	return embed_batch(batch, embeddings_tensor,frame_size)
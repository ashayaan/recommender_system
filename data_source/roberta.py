# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-05 16:26:15
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-06 12:12:47

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
from fairseq.data.data_utils import collate_tokens

batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base').to(device) 	#Loading the Bert pretrained model


#using yield to return list and continue the loop from the same point
def get_batch(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n] 

def extract_features(batch,ids,fs):
	#Converting 1d tensor to 2d padded tensor
	batch = collate_tokens([roberta.encode(sent) for sent in batch], pad_idx=1).to(device)
	batch = batch[:, :512]
	features = roberta.extract_features(batch)
	pooled_features = F.avg_pool2d(features, (features.size(1), 1)).squeeze()
	for i in range(pooled_features.size(0)):
		fs[ids[i]] = pooled_features[i].detach().cpu().numpy()

if __name__ == '__main__':
	roberta.eval()
	omdb = json.load(open('../data/parsed/omdb.json'))
	tmdb = json.load(open('../data/parsed/tmdb.json'))
	
	plots = []
	for i in tmdb.keys():
		omdb_plot = omdb[i]['omdb'].get('Plot', '')
		tmdb_plot = tmdb[i]['tmdb'].get('overview', '')
		plot = tmdb_plot + ' ' + omdb_plot
		plots.append((i, plot, len(plot)))

	
		
	plots = list(sorted(plots, key=lambda x: x[2]))
	print(plots[1])
	plots = list(filter(lambda x: x[2] > 4, plots))


	ids = [i[0] for i in plots]
	plots = [i[1] for i in plots]
	plots = list(get_batch(plots, batch_size))
	ids = list(get_batch(ids, batch_size))
	
	features = {} 
	count = 0
	for batch, ids in zip(plots[::-1], ids[::-1]):
		if count > 3:
			break
		extract_features(batch, ids, features)
		count+=1

	roberta_data = pd.DataFrame(features).T
	roberta_data.index = roberta_data.index.astype(int)
	roberta_data = roberta_data.sort_index()
	roberta_data.to_csv('../data/roberta.csv',index=True,index_label='idx')
# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-07 11:22:28
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-07 12:44:33

import torch
import pandas as pd 
import numpy as np
import json
from ppca import PPCA
import pickle

if __name__ == '__main__':
	roberta_features = pd.read_csv('../data/roberta.csv')
	roberta_features = roberta_features.set_index('idx')

	mca_features = pd.read_csv('../data/mca.csv')
	mca_features = mca_features.set_index('idx')

	pca_features = pd.read_csv('../data/pca.csv')
	pca_features = pca_features.set_index('idx')

	links = pd.read_csv('../data/ml-20m/links.csv')

	df = pd.concat([roberta_features, mca_features, pca_features], axis=1)
	ppca = PPCA() 
	ppca.fit(data=df.values.astype(float),d=128,verbose=True)
	print(ppca.var_exp)

	transformed = ppca.transform()
	films_dict = dict([(k, torch.tensor(transformed[i]).float()) for k, i in zip(df.index, range(transformed.shape[0]))])
	pickle.dump(films_dict, open('../data//ml20_pca128.pkl', 'wb'))
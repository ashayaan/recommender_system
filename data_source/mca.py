# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-06 10:33:24
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-12 14:39:08

'''
Multiple correspondence analysis is only applied to the 
categorical fields of the data
'''


import json
import pandas as pd
import numpy as np
import prince
import itertools
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

def generateOneHotEncoding(arr, name, categories):
	return dict((name+i, i in arr) for i in categories)

def applyOneHot(records, type, name, categories):
	for row in records.keys():
		records[row] = {**records[row], **generateOneHotEncoding(records[row][type], name, categories)}
		del records[row][type]
	return records

def extractFeatures(records, type, take):
	res = {i: {} for i in records.keys()}
	for row in records.keys():
		for col in records[row][type].keys():
			if col in take:
				res[row][col] = records[row][type][col]
	return res

def splitFeatures(records, split):
	for row in records.keys():
		for col in split:
			records[row][col] = tuple(records[row][col].split(', '))
	return records

def fillMissing(data):
	data.Rated = data.Rated.fillna('Not Rated')
	data.Rated[data.Rated == 'N/A'] = 'Not Rated'
	data.Production.fillna('-')
	data.Production[data.Production == 'N/A'] = '-'
	data.Production = data.Production.fillna('-')
	data.Production[data.Production == 'NaN'] = '-'
	data.Production[data.Production.isna()] = '-'
	data.Director.fillna('-')
	data.Director[data.Director == 'N/A'] = '-'
	return data

if __name__ == '__main__':
	omdb = json.load(open('../data/parsed/omdb.json'))
	tmdb = json.load(open('../data/parsed/tmdb.json'))
	features = {'omdb': ['Rated', 'Director', 'Genre', 'Language', 'Country', 'Type', 'Production'],}
	extracted_features = extractFeatures(omdb, 'omdb', features['omdb'])
	extracted_features = splitFeatures(extracted_features,['Country', 'Language', 'Genre'])
	df = pd.DataFrame.from_dict(extracted_features).T
	
	genres_features = list(set(itertools.chain(*tuple(df.Genre))))
	language_features = pd.Series(list(itertools.chain(*df.Language))).value_counts()[:30].index
	countries_features = pd.Series(list(itertools.chain(*df.Country))).value_counts()[:30].index

	extracted_features = applyOneHot(extracted_features, 'Genre', 'g_', genres_features)
	extracted_features = applyOneHot(extracted_features, 'Country', 'c_', countries_features)
	extracted_features = applyOneHot(extracted_features, 'Language', 'l_', language_features)
	df = pd.DataFrame.from_dict(extracted_features).T

	df = fillMissing(df)
	# df = df.head(6000)
	
	#Multiple Correspondence analysis
	mca = prince.MCA(n_components=16,n_iter=20,copy=True,check_input=True,engine='auto',)
	print(time.ctime())
	mca.fit(df)
	print(time.ctime())
	
	transformed = mca.transform(df)
	transformed.to_csv('../data/mca.csv',index=True,index_label='idx')
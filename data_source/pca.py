# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-06 14:42:30
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-07 00:00:50

import pandas as pd
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd
from ppca import PPCA
import time


def extractFeatures(records, type, take):
	res = {i: {} for i in records.keys()}
	for row in records.keys():
		for col in records[row][type].keys():
			if col in take:
				res[row][col] = records[row][type][col]
	return res

def extractRatings(records):
	res = records.copy()
	for i in res.keys():
		for rating in res[i]['Ratings']:
			res[i][rating['Source']] = rating['Value']
		del res[i]['Ratings']
	return res

def replaceForwardSlash(x):
	try:
		return float(x.split('/')[0])
	except:
		return x

def replacePercentage(x):
	try:
		return float(x.split('%')[0])
	except Exception as e:
		return x

def fixData(data):
	for col in data.columns:
		data[col].loc[data[col] == 'N/A'] = np.nan
	data['budget'] = data['budget'].replace(to_replace=0, value=np.nan)
	data['Internet Movie Database'].loc[data['Internet Movie Database'].notnull()] = \
	data['Internet Movie Database'].loc[data['Internet Movie Database'].notnull()].apply(lambda x: x.split('/')[0])
	data['Metacritic'].loc[data['Metacritic'].notnull()] = \
	data['Metacritic'].loc[data['Metacritic'].notnull()].apply(lambda x: int(x.split('/')[0]))
	data['Rotten Tomatoes'].loc[data['Rotten Tomatoes'].notnull()] = \
	data['Rotten Tomatoes'].loc[data['Rotten Tomatoes'].notnull()].apply(lambda x: float(x.replace('%', '')))
	data['revenue'] = data['revenue'].replace(to_replace=0, value=np.nan)
	data['Year'].loc[data['Year'].notnull()] = data['Year'].loc[data['Year'].notnull()].apply(lambda x: int(x.replace('–', '')[0]))
	data['imdbVotes'].loc[data['imdbVotes'].notnull()] = data['imdbVotes'].loc[data['imdbVotes'].notnull()].apply(lambda x: int(x.replace(',', ''))) 
	return data

if __name__ == '__main__':
	omdb = json.load(open('../data/parsed/omdb.json'))
	tmdb = json.load(open('../data/parsed/tmdb.json'))
	
	numerical_features = {'omdb': ['Year', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes'],
						'tmdb': ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']}

	omdb_numerical = extractFeatures(omdb,'omdb',numerical_features['omdb'])
	tmdb_numerical = extractFeatures(tmdb,'tmdb',numerical_features['tmdb'])
	data = dict([(i, {**omdb_numerical[i], **tmdb_numerical[i]}) for i in omdb_numerical.keys()])
	data = extractRatings(data)
	# data = dict([(i,{**omdb_numerical[i],**tmdb_numerical[i]}) for i in omdb_numerical.keys()])
	df = pd.DataFrame.from_dict(data).T
	df.replace('N/A',np.nan,inplace=True)
	df.to_pickle('temp.pkl')
	df = fixData(df)
	
	ppca = PPCA()
	print (time.ctime())
	ppca.fit(df.values.astype(float), d=16,verbose=True)
	print (time.ctime())

	transformed = ppca.transform()
	transformed = pd.DataFrame(transformed)
	transformed['idx'] = pd.Series(list(omdb.keys()))
	transformed = transformed.set_index('idx')
	transformed.head()
	transformed.to_csv('../data/pca.csv',index=True,index_label='idx')

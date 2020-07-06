# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2020-07-06 14:42:30
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-07-06 14:47:17

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd
import ppca
import numpy as np


if __name__ == '__main__':
	omdb = json.load(open("../../../../data/parsed/omdb.json", "r") )
	tmdb = json.load(open("../../../../data/parsed/tmdb.json", "r") )

	
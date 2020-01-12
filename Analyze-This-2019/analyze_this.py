# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:50:12 2019

@author: PASHUPATI-PC
"""

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline

url_train = './Dataset/development_dataset.csv' #traing dataset
url_dict = './Dataset/Data_Dictionary.xlsx' #data dictionary

## making dataframe

df_train = pd.read_csv(url_train)
df_dict = pd.read_excel(url_dict)

df_dict.head()

df_train.head(20)

def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)

df_with_any_null_values = df_train[df_train.columns[df_train.isnull().any()].tolist()]

get_percentage_missing(df_with_any_null_values)

del df_train['VAR9']
del df_train['VAR17']

df_train = df_train.replace(to_replace = '.', value =np.nan)

features = list(df_dict['Variable Name'][:21])
all_columns = df_train.columns.tolist()
features

features.remove('Internal Revolve') #index
features.remove('External_rev_rate') #target
print(len(features))

len(all_columns)

df_var = pd.DataFrame(list(zip(all_columns,features)), 
               columns =['column', 'description'])
df_var.head()

#Relevent columns
columns = all_columns
#type(columns)
columns.remove('VAR1') #index
columns.remove('VAR21') #target
columns.remove('VAR14') #categorical
#columns

#Required Imports

from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from sklearn.impute import SimpleImputer

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

# Techinques
# 1 - df.fillna()
# 2 - impute using sklearn - mean,meadian,mode
# 3 - interpolation 
# 4 - LDA analysis

####   impute using sklearn - mean,meadian,mode 
imputer_mean = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_median = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_mode = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

df_train.columns

# Imputing column 1 to 11 using mean
imputer_mean = imputer_mean.fit(df_train.iloc[:, 1:12].values)
df_train.iloc[:, 1:12] = imputer_mean.fit_transform(df_train.iloc[:, 1:12].values)

# Imputing column 13 and 14 using mean
imputer_mean = imputer_mean.fit(df_train.iloc[:,13:14].values)
df_train.iloc[:,13:14] = imputer_mean.transform(df_train.iloc[:,13:14].values)

# Imputing VAR 20 using mean
flag20 = df_train['VAR20'].as_matrix().reshape(-1,1)
imputer_mode = imputer_mode.fit(flag20)
flag20 = imputer_mean.transform(flag20)
df_train['VAR20'] = flag20











# IMPORTS
import re
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import lightgbm as gbm

# CONSTANTS
TRAINSET_HEADERS = ['application_key', 'mvar1', 'mvar2', 'mvar3', 'mvar4', 'mvar5', 'mvar6',
       'mvar7', 'mvar8', 'mvar9', 'mvar10', 'mvar11', 'mvar12', 'mvar13',
       'mvar14', 'mvar15', 'mvar18', 'mvar19',
       'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27',
       'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34',
       'mvar35', 'mvar36', 'mvar37', 'mvar38', 'mvar40', 'mvar41',
       'mvar42', 'mvar43', 'mvar44', 'mvar45', 'mvar47',
       'default_ind']
TESTSET_HEADERS = TRAINSET_HEADERS[:-1]
LABEL = TRAINSET_HEADERS[-1]
KEY = TRAINSET_HEADERS[0]
CATEGORICAL_DATA = 'mvar47'

# LOADING FILES
org_train = pd.read_csv('data/Training_dataset_Original.csv')
org_test = pd.read_csv('data/Evaluation_dataset.csv')

#Converting categorical to Numerical Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
org_train[CATEGORICAL_DATA] = le.fit_transform(org_train[CATEGORICAL_DATA])
org_test[CATEGORICAL_DATA] = le.fit_transform(org_test[CATEGORICAL_DATA])

# REPLACING missing, na, N/A with numpy nan object
for col in org_train.columns:
    if org_train[col].dtypes == object:
        org_train.loc[org_train[col] == 'na', col] = np.nan
        org_train.loc[org_train[col] == 'N/A', col] = np.nan
        org_train.loc[org_train[col] == 'missing', col] = np.nan
        org_train[col] = pd.to_numeric(org_train[col])
        
for col in org_test.columns:
    if org_test[col].dtypes == object:
        org_test.loc[org_test[col] == 'na', col] = np.nan
        org_test.loc[org_test[col] == 'N/A', col] = np.nan
        org_test.loc[org_test[col] == 'missing', col] = np.nan
        org_test[col] = pd.to_numeric(org_test[col])

# Saving the labels and keys for future
labels = pd.Series(org_train[LABEL].values)
key = org_test[KEY]
org_train.drop([KEY], axis = 1, inplace = True)
org_test.drop([KEY], axis = 1, inplace = True)

######################################################################################################

"""Impute missing values with k nearest classifier."""
import sys
import numpy as np
import pandas as pd
from sklearn import neighbors


class Imputer:
    """Imputer class."""

    def _fit(self, X, column, k=10, is_categorical=False):
        """Fit a knn classifier for missing column.

        - Args:
                X(numpy.ndarray): input data
                column(int): column id to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                clf: trained k nearest neighbour classifier
        """
        clf = None
        if not is_categorical:
            clf = neighbors.KNeighborsRegressor(n_neighbors=k)
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        # use column not null to train the kNN classifier
        missing_idxes = np.where(pd.isnull(X[:, column]))[0]
        if len(missing_idxes) == 0:
            return None
        X_copy = np.delete(X, missing_idxes, 0)
        X_train = np.delete(X_copy, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_train[:, col_id]))[0]
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_train[col_missing_idxes, col_id] = col_mean[col_id]
        y_train = X_copy[:, column]
        # fit classifier
        clf.fit(X_train, y_train)
        return clf

    def _transform(self, X, column, clf, is_categorical):
        """Impute missing values.

        - Args:
                X(numpy.ndarray): input numpy ndarray
                column(int): index of column to be imputed
                clf: pretrained classifier
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X(pandas.dataframe): imputed dataframe
        """
        missing_idxes = np.where(np.isnan(X[:, column]))[0]
        X_test = X[missing_idxes, :]
        X_test = np.delete(X_test, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        # fill missing values in each column with current col_mean
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_test[:, col_id]))[0]
            # if no missing values for current column
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_test[col_missing_idxes, col_id] = col_mean[col_id]
        # predict missing values
        y_test = clf.predict(X_test)
        X[missing_idxes, column] = y_test
        return X

    def knn(self, X, column, k=10, is_categorical=False):
        """Impute missing value with knn.

        - Args:
                X(pandas.dataframe): dataframe
                column(str): column name to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X_imputed(pandas.dataframe): imputed pandas dataframe
        """
        X, column = self._check_X_y(X, column)
        clf = self._fit(X, column, k, is_categorical)
        if clf is None:
            return X
        else:
            X_imputed = self._transform(X, column, clf, is_categorical)
            return X_imputed

    def _check_X_y(self, X, column):
        """Check input, if pandas.dataframe, transform to numpy array.

        - Args:
                X(ndarray/pandas.dataframe): input instances
                column(str/int): column index or column name
        - Returns:
                X(ndarray): input instances
        """
        column_idx = None
        if isinstance(X, pd.core.frame.DataFrame):
            if isinstance(column, str):
                # get index of current column
                column_idx = X.columns.get_loc(column)
            else:
                column_idx = column
            X = X.as_matrix()
        else:
            column_idx = column
        return X, column_idx

train_len = len(org_train)
label = org_train[LABEL]
temp_org_train = org_train.drop([LABEL], axis = 1)
temp_data = pd.concat([temp_org_train, org_test], axis = 0)
saveit = temp_data['mvar47']
data = temp_data.drop(['mvar47'], axis = 1)

save_series = dict()

missing_col = ['mvar1', 'mvar6', 'mvar7', 'mvar8', 'mvar9', 'mvar18', 'mvar25', 'mvar26', 'mvar27', 'mvar37']
impute = Imputer()
for var in missing_col:
    data_as_array = impute.knn(X=data, column=var) # default 10nn
    kuch_bhi = pd.DataFrame(data_as_array, columns = data.columns)
    save_series[var] = kuch_bhi[var]
data = temp_data.drop(missing_col, axis = 1)

data = data.assign(mvar1=save_series['mvar1'].values)
data = data.assign(mvar6=save_series['mvar6'].values)
data = data.assign(mvar7=save_series['mvar7'].values)
data = data.assign(mvar8=save_series['mvar8'].values)
data = data.assign(mvar9=save_series['mvar9'].values)
data = data.assign(mvar18=save_series['mvar18'].values)
data = data.assign(mvar25=save_series['mvar25'].values)
data = data.assign(mvar26=save_series['mvar26'].values)
data = data.assign(mvar27=save_series['mvar27'].values)
data = data.assign(mvar37=save_series['mvar37'].values)
    
tryout = data.copy()
tryout = tryout.assign(mvar47=saveit.values)

org_train = tryout[:train_len]
org_train = org_train.assign(default_ind=label.values)
org_test = tryout[train_len:]

######################################################################################################


X = org_train.ix[:, org_train.columns != LABEL]
y = org_train.ix[:, org_train.columns == LABEL]
testset = org_test.ix[:]


# Undersampling is done 20 times and the probabilites are averaged out
y_pred_list = list()
for i in range(20):
    # Number of data points in the minority class
    number_records_fraud = len(org_train[org_train[LABEL] == 1])
    fraud_indices = np.array(org_train[org_train[LABEL] == 1].index)
    normal_indices = org_train[org_train[LABEL] == 0].index

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
    random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
    under_sample_data = org_train.iloc[under_sample_indices,:]
    X_undersample = under_sample_data.ix[:, under_sample_data.columns != LABEL]
    y_undersample = under_sample_data.ix[:, under_sample_data.columns == LABEL]

    cv_params = {
        'learning_rate': 0.01,
        'max_depth': 20,
        'min_child_weight': 5,
        'min_child_samples': 3,
        'objective': 'binary',
        'metric':'binary_logloss',
        'min_split_gain': 0.5,
        'scale_pos_weight': 1,
        'num_leaves': 8,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.4,
        'max_bin': 50,
        'subsample': 0.6,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.9,
        'reg_lambda': 0.1
        }

    evals_result = {}  # to record eval results for plotting
    gbm_tr = gbm.Dataset(X_undersample, label=y_undersample.values.ravel())
    INITIAL_MODEL = gbm.train(cv_params,
                    gbm_tr,
                    num_boost_round=3200,
                    evals_result=evals_result,
                    verbose_eval=1)


    gbm_ts = gbm.Dataset(testset)
    y_pred=INITIAL_MODEL.predict(testset)
    y_pred_list.append(y_pred)


y_pred = [sum(x) for x in zip(*y_pred_list)]
y_pred = [x / 20 for x in y_pred]






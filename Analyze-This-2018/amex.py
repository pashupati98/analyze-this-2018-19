# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:46:07 2018

@author: user
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Training_dataset_Original.csv')

leaddataset = pd.read_csv('Leaderboard_dataset.csv')

#relacing na and missing 
dataset.replace('na', np.nan, inplace = True)
dataset.replace('missing', np.nan, inplace = True)

leaddataset.replace('na', np.nan, inplace = True)
leaddataset.replace('missing', np.nan, inplace = True)

#deleting useless feature
del dataset['mvar11']
del dataset['mvar23']
del dataset['mvar24']
del dataset['mvar31']

del leaddataset['mvar11']
del leaddataset['mvar23']
del leaddataset['mvar24']
del leaddataset['mvar31']

# splitting X and Y
X = dataset.iloc[:, 1:44].values
y = dataset.iloc[:, 44].values

tX = leaddataset.iloc[:, 1:44].values


# hendelling missing values 
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer3 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

#column 0 and 1
imputer1 = imputer1.fit(X[:, 0:2])
X[:, 0:2] = imputer1.transform(X[:, 0:2])
#column 2,3,4
imputer3 = imputer3.fit(X[:, 2:5])
X[:, 2:5] = imputer3.transform(X[:, 2:5])
#column 5 to 13
imputer1 = imputer1.fit(X[:, 5:14])
X[:, 5:14] = imputer1.transform(X[:, 5:14])
#column 14 to 18
imputer3 = imputer3.fit(X[:, 14:19])
X[:, 14:19] = imputer3.transform(X[:, 14:19])
#column 19 to 39
imputer1 = imputer1.fit(X[:, 19:40])
X[:, 19:40] = imputer1.transform(X[:, 19:40])
#column 40,41
imputer3 = imputer3.fit(X[:, 40:42])
X[:, 40:42] = imputer3.transform(X[:, 40:42])

# leaderboard dataset

#column 0 and 1
imputer1 = imputer1.fit(tX[:, 0:2])
tX[:, 0:2] = imputer1.transform(tX[:, 0:2])
#column 2,3,4
imputer3 = imputer3.fit(tX[:, 2:5])
tX[:, 2:5] = imputer3.transform(tX[:, 2:5])
#column 5 to 13
imputer1 = imputer1.fit(tX[:, 5:14])
tX[:, 5:14] = imputer1.transform(tX[:, 5:14])
#column 14 to 18
imputer3 = imputer3.fit(tX[:, 14:19])
tX[:, 14:19] = imputer3.transform(tX[:, 14:19])
#column 19 to 39
imputer1 = imputer1.fit(tX[:, 19:40])
tX[:, 19:40] = imputer1.transform(tX[:, 19:40])
#column 40,41
imputer3 = imputer3.fit(tX[:, 40:42])
tX[:, 40:42] = imputer3.transform(tX[:, 40:42])

# 42, categorical

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 42] = labelencoder_X.fit_transform(X[:, 42])
onehotencoder = OneHotEncoder(categorical_features = [42])
X = onehotencoder.fit_transform(X).toarray()

# leader board dataset

tX[:, 42] = labelencoder_X.fit_transform(tX[:, 42])
onehotencoder1 = OneHotEncoder(categorical_features = [42])
tX = onehotencoder1.fit_transform(tX).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Predicting the leader board results
ans = classifier.predict(tX)









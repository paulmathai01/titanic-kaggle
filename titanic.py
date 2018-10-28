import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

df_train=pd.read_csv('train.csv',sep=',')
df_test=pd.read_csv('test.csv',sep=',')
df_data = df_train.append(df_test) # The entire data: train + test.

PassengerId = df_test['PassengerId']
Submission=pd.DataFrame()
Submission['PassengerId'] = df_test['PassengerId']

print(df_train.shape)
print("----------------------------")
print(df_test.shape)

df_train.columns

df_data.info()

print(pd.isnull(df_data).sum())

df_train.describe()

df_test.describe()

df_train.head(5)

df_train.tail(5)

NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Fare']

# create test and training data
data_to_train = df_train[NUMERIC_COLUMNS].fillna(-1000)
y=df_train['Survived']
X=data_to_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)

from sklearn.svm import LinearSVC

clf = SVC()
clf.fit(X_train, y_train)
linear_svc = LinearSVC()

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))

test = df_test[NUMERIC_COLUMNS].fillna(-1000)
Submission['Survived']=clf.predict(test)
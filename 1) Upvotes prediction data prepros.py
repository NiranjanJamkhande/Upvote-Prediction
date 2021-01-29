# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:51:29 2020

@author: Admin
"""


#### Upvotes prediction Analysis  ##########

import pandas as pd
import numpy as np


train_o = pd.read_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\train.csv')
test_o = pd.read_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\test.csv')
sub = pd.read_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sample_submission.csv')

### dropping unnecessory columns from the data.
train = train_o.drop(columns = ['ID','Username'])
test = test_o.drop(columns = ['ID','Username'])

train.corr()
## there is no multy collinearity

### testing missing values.
null_col = train.columns[train.isnull().any()]
null_col
train[null_col].isnull().sum()

null_col = test.columns[test.isnull().any()]
null_col
test[null_col].isnull().sum()


###### 

train.info()
test.info()

train['Tag'].value_counts()
test['Tag'].value_counts()

## dummy the features
train = pd.get_dummies(train, drop_first = True)
test = pd.get_dummies(test, drop_first = True)
### 


X_train = train.drop(columns = ['Upvotes'])
y_train = train.iloc[:,3]
X_test = test

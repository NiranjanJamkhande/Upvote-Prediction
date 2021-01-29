# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:14:55 2020

@author: Admin
"""

### K NN
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit( X_train , y_train )
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]




submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-KNN_7.csv')

#### Linear regg
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-lin regg.csv')

####  Ridge

from sklearn.linear_model import Ridge


clf = Ridge(alpha=2)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-ridge regg 2.csv')

#### Lasso

from sklearn.linear_model import Lasso

clf = Lasso(alpha=3)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-lasso regg 2.csv')


### Elastic net
from sklearn.linear_model import ElasticNet



clf = ElasticNet(alpha=2, l1_ratio=0.6)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-en.csv')


#####################################################################################


###################### SGD Regressor ################
from sklearn.linear_model import SGDRegressor
sgdReg = SGDRegressor(random_state=2019)
sgdReg.fit(X_train,y_train)
y_pred = sgdReg.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-sgd.csv')



################ decision tree

from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf2 = clf.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-dt.csv')

#########  Random Forest  ##########

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(max_depth = 8)

model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-rf.csv')



###### XGB  ###


from xgboost import XGBRegressor
clf = XGBRegressor()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-xgb.csv')


################ Tunning XG Boost ##################################
from sklearn.metrics import r2_score
lr_range = [0.001, 0.01, 0.1, 0.2,0.25, 0.3]
#n_est_range = [10,20,30,50,100]
md_range = [2,4,6,8,10]

parameters = dict(learning_rate=lr_range,
#                  n_estimators=n_est_range,
                  max_depth=md_range)

from sklearn.model_selection import GridSearchCV
clf = XGBRegressor(random_state=1211)
cv = GridSearchCV(clf, param_grid=parameters,
                  scoring='r2')

cv.fit(X_train,y_train)

print(cv.best_params_)

print(cv.best_score_)

############################################################################



############### Gradient boosting  #####


from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(random_state=1200)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-gb.csv')


#####################  Voting   ###############


from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(max_depth = 8)


from xgboost import XGBRegressor
xgb = XGBRegressor()


from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(random_state=1200)



from sklearn.ensemble import VotingRegressor






# Average
# Voting = VotingRegressor(estimators=[('DT',dtr),('LR',lr),('SV',svr)])

#OR Weighted Average
Voting = VotingRegressor(estimators=[('RF',model_rf),('XGB',xgb),('GB',gb)],
                                     weights=np.array([0.2,0.5,0.3]))



Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-voting 1.csv')

#### ANN



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

columns_to_scale = ['Reputation', 'Views']
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])





from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(7,4,2,1),random_state=2018)
mlp.fit( X_train , y_train )
y_pred = mlp.predict(X_test)



#################### Stacking   #######


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=1200)


from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=1200,max_depth=5)

from sklearn.linear_model import SGDRegressor
sgdReg = SGDRegressor(random_state=1200, alpha=0.4)


models_considered = [('XGB', XGB),
                     ('DTREE', dtree),
                     ('SGD',sgdReg) ]
                    
from xgboost import XGBRegressor
XGB = XGBRegressor(random_state=1200)


from sklearn.ensemble import StackingRegressor
stack = StackingRegressor(estimators = models_considered,
                           final_estimator=gbr, verbose = 3 )
                          

stack.fit(X_train,y_train)
y_pred = stack.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-stacking 2.csv')


#################################  Bagging  ####

from sklearn.ensemble import BaggingRegressor

# Default: Tree Regressor
br = BaggingRegressor(random_state=1211,oob_score=True,
                            max_features = X_train.shape[1],
                            max_samples=X_train.shape[0], verbose = 3 )
                           

br.fit( X_train , y_train )
y_pred = br.predict(X_test)
y_pred = pd.DataFrame(y_pred)

y_pred['Upvotes'] = y_pred[0]
y_pred = y_pred.iloc[:,1]


submission = pd.concat([test_o['ID'],y_pred],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\Upvote prediction\\sub-bagging 4.csv')



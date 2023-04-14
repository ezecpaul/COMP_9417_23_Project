import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA
sys.path.append('.') # point to the root directory
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')

### Read and cleaned data and generate the PCA dataframes:
from utils import util
from feature_select import FS

data_file = os.path.join(data_dir, 'training.csv')

df = pd.read_csv(data_file)
data = util.clean_outliers(df, drop_outliers = True )
X_train, X_test, y_train, y_test = util.prepare_data(data, test_split_ratio=0.2)

pca, top_features = FS.pca(X_train, explain_threshold=0.99)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)

##==================Hyperparameter tuning=========================
# Specify model 
forest = RandomForestRegressor()

param_grid = {
    'n_estimators': [10, 50, 100,200],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [ None, 'sqrt', 'log2']
}

## FOR PCA DATA
grid_search1 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search1.fit(X_train_PCA, y_train.iloc[:,0])

grid_search2 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search2.fit(X_train_PCA, y_train.iloc[:,1])

grid_search3 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search3.fit(X_train_PCA, y_train.iloc[:,2])

grid_search4 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search4.fit(X_train_PCA, y_train.iloc[:,3])

grid_search5 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search5.fit(X_train_PCA, y_train.iloc[:,4])


## FOR NON-PCA DATA

grid_search1 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search1.fit(X_train, y_train.iloc[:,0])

grid_search2 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search2.fit(X_train, y_train.iloc[:,1])

grid_search3 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search3.fit(X_train, y_train.iloc[:,2])

grid_search4 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search4.fit(X_train, y_train.iloc[:,3])

grid_search5 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search5.fit(X_train, y_train.iloc[:,4])


### =======Modify model parameters and list the most important features of each output feature:========


#### For PCA data set: 
model1_PCA = RandomForestRegressor(max_depth=20,
                               min_samples_leaf=2,
                               min_samples_split=2,
                               n_estimators=100)
model1_PCA.fit(X_train_PCA, y_train.iloc[:,0])
top_features_count = 15
importance = model1_PCA.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f1_PCA:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")


model2_PCA = RandomForestRegressor(max_depth=25,
                                   max_features="sqrt",
                                   min_samples_leaf=1,
                                   min_samples_split=2,
                                   n_estimators=50,
                                 )
model2_PCA.fit(X_train_PCA, y_train.iloc[:,1])
top_features_count = 15
importance = model2_PCA.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f2_PCA:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model3_PCA = RandomForestRegressor(max_depth=15,
                               min_samples_leaf=1,
                               min_samples_split=2,
                               n_estimators=200)
model3_PCA.fit(X_train_PCA, y_train.iloc[:,2])
top_features_count = 15
importance = model3_PCA.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f3_PCA:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model4_PCA = RandomForestRegressor(max_depth=15,
                               min_samples_leaf=1,
                               min_samples_split=2,
                               n_estimators=100)
model4_PCA.fit(X_train_PCA, y_train.iloc[:,3])
top_features_count = 15
importance = model4_PCA.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f4:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model5_PCA = RandomForestRegressor(max_depth=20,
                                   max_features="sqrt",
                               min_samples_leaf=1,
                               min_samples_split=2,
                               n_estimators=200)
model5_PCA.fit(X_train_PCA, y_train.iloc[:,4])
top_features_count = 15
importance = model5_PCA.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f5_PCA:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")


#### For non-PCA data set:

model1 = RandomForestRegressor(max_depth=15,
                               max_features="sqrt",
                               min_samples_leaf=1,
                               min_samples_split=5,
                               n_estimators=10,)
model1.fit(X_train, y_train.iloc[:,0])
top_features_count = 15
importance = model1.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f1:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")




model2 = RandomForestRegressor(max_features="sqrt",
                               min_samples_leaf=5,
                               min_samples_split=5,
                               n_estimators=100,
                               random_state=42)
model2.fit(X_train, y_train.iloc[:,1])
top_features_count = 15
importance = model2.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f2:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model3 = RandomForestRegressor(max_depth=15,
                               min_samples_leaf=1,
                               min_samples_split=2,
                               n_estimators=100)
model3.fit(X_train, y_train.iloc[:,2])
top_features_count = 15
importance = model3.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f3:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model4 = RandomForestRegressor(max_depth=25,
                               max_features="sqrt",
                               min_samples_leaf=1,
                               min_samples_split=2,
                               n_estimators=300)
model4.fit(X_train, y_train.iloc[:,3])
top_features_count = 15
importance = model4.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f4:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")



model5 = RandomForestRegressor(max_depth=20,
                               min_samples_leaf=2,
                               min_samples_split=5,
                               n_estimators=100)
model5.fit(X_train, y_train.iloc[:,4])
top_features_count = 15
importance = model5.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
top_feature_indices = sorted_indices[:top_features_count]
print("Top 10 important features of f5:")
for i, index in enumerate(top_feature_indices):
    print(f"{i + 1}. Feature {X_train.columns[index]} (importance: {importance[index]})")


#====================## Prediction and calculate the RMSE value:===================

#### For PCA data set:
erors_test_PCA= []
y_pred_1 = model1_PCA.predict(X_test_PCA).reshape(1,-1).T
y_pred_2 = model2_PCA.predict(X_test_PCA).reshape(1,-1).T
y_pred_3 = model3_PCA.predict(X_test_PCA).reshape(1,-1).T
y_pred_4 = model4_PCA.predict(X_test_PCA).reshape(1,-1).T
y_pred_5 = model5_PCA.predict(X_test_PCA).reshape(1,-1).T

erors_test_PCA.append(mean_squared_error(y_test.values[:,0].reshape(1,-1).T, y_pred_1, squared=False))
erors_test_PCA.append(mean_squared_error(y_test.values[:,1].reshape(1,-1).T, y_pred_2, squared=False))
erors_test_PCA.append(mean_squared_error(y_test.values[:,2].reshape(1,-1).T, y_pred_3, squared=False))
erors_test_PCA.append(mean_squared_error(y_test.values[:,3].reshape(1,-1).T, y_pred_4, squared=False))
erors_test_PCA.append(mean_squared_error(y_test.values[:,4].reshape(1,-1).T, y_pred_5, squared=False))

print(f'MCRMSE/average-RMSE (PCA DATA): {sum(erors_test_PCA)/5}')


#### For non-PCA data set:

erors_test= []
y_pred_1 = model1.predict(X_test).reshape(1,-1).T
y_pred_2 = model2.predict(X_test).reshape(1,-1).T
y_pred_3 = model3.predict(X_test).reshape(1,-1).T
y_pred_4 = model4.predict(X_test).reshape(1,-1).T
y_pred_5 = model5.predict(X_test).reshape(1,-1).T

erors_test.append(mean_squared_error(y_test.values[:,0].reshape(1,-1).T, y_pred_1, squared=False))
erors_test.append(mean_squared_error(y_test.values[:,1].reshape(1,-1).T, y_pred_2, squared=False))
erors_test.append(mean_squared_error(y_test.values[:,2].reshape(1,-1).T, y_pred_3, squared=False))
erors_test.append(mean_squared_error(y_test.values[:,3].reshape(1,-1).T, y_pred_4, squared=False))
erors_test.append(mean_squared_error(y_test.values[:,4].reshape(1,-1).T, y_pred_5, squared=False))

print(f'MCRMSE/average-RMSE (NON_PCA DATA): {sum(erors_test)/5}')
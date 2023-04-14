import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
saved_searchCV_dir = os.path.join(ROOT_DIR, 'svr_searchCV_pkl')
data_dir = os.path.join(ROOT_DIR, 'data')

# Read in train Dataset 
file = os.path.join(data_dir, 'training.csv')
df =  pd.read_csv(file)

#import utility/helper function class
from utils import util

# clean data and drop or replace outliers- set drop_outlers to True to drop or False (default) to replace them
data = util.clean_outliers(df, drop_outliers = True )

# set test_split_ratio =0.0 ( X_test and y_test becomes None) when preparing for Final Submission Otherwise set to (0.1 or 0.2): default-0.2
X_train, _, y_train, _ = util.prepare_data(data, test_split_ratio=0.0)

t_file = os.path.join(data_dir, 'sorted_test.csv')
X_test = pd.read_csv(t_file)
X_test.drop(['PIDN', 'Depth'], axis=1, inplace=True) 


svr_dir = os.path.join(ROOT_DIR, 'svr_searchCV_pkl') # directory where GridsearchCV files are stored
files = ['Ca.pkl', 'P.pkl', 'pH.pkl', 'Sand.pkl', 'Soc.pkl']
searchCVs = list()
print('\nPreparing submission file...')
for i in range(5):
    cv_file = joblib.load(os.path.join(svr_dir, files[i]))
    searchCVs.append(cv_file)
labels = list(y_train.columns)
y_pred = np.zeros((X_test.shape[0], 5))
for i in range(5):
    X, y = np.array(X_train), np.array(y_train[labels[i]])
    model = svm.SVR(**searchCVs[i].best_params_).fit(X, y)
    y_pred[:, i] = model.predict(X_test)
    
out_file = os.path.join(data_dir, 'sample_submission.csv')
util.submission(out_file, y_pred)
print('\nDone!')
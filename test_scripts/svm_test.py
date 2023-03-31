import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')
data_files_dir = os.path.join(ROOT_DIR, 'data_pkl')
model_file = os.path.join(saved_model_dir, 'svr_model.pkl')  
svr_metrics_dir =os.path.join(model_file, 'svr_metrics')                        

from utils import util
from models.svm import SVR_Model # use this when tuning is not done yet.

# step 1:
# read data sets (X_train, y_train, X_test, y_test)
X_train = pd.read_csv(os.path.join(data_files_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_files_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_files_dir, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(data_files_dir, 'y_test.csv'))


# step 2:
# Read best parameter file "model_pkl\svr_metrics\svr_best_metric.csv"
# example:best params = {'C': 10000.0, 'coef0':0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}

#Step 3: Use best parameter to build define and build model
model = svm.SVR(C= 15000.0, 
                coef0=0.1, 
                degree= 3, 
                gamma= 'auto', 
                kernel= 'poly')

# step 4: Fit model to different Datasets created in data_pkl ( VIF_dataset, Raw_dataset, PCA dataset, etc)
pred = util.fit_predict(model, 
                      X_train, 
                      y_train, 
                      X_test)

# Step 5: Plot graghs and compare metrics
RMSEs, Ave_RMSE = util.average_RMSE(y_test, pred)
print(f'\nColumns [Ca, P, pH, SOC , Sand] RMSEs: \n {RMSEs} \n Average RMSE: {Ave_RMSE}')

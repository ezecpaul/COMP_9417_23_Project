import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')
data_files_dir = os.path.join(ROOT_DIR, 'data_pkl')
searchCV_file = os.path.join(saved_model_dir, 'svr_searchCV.pkl')  

from utils import util
X_train, X_test, y_train, y_test = util.process_data('training.csv')

from feature_selection import Fselect as fs
features = fs.spearman_corr(X_train, y_train, 100)
print(len(features))
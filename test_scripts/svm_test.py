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


import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import time

sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')
processed_data_dir = os.path.join(ROOT_DIR, 'data_pkl')
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')


# from utils import util
# X_train, X_test, y_train, y_test = util.prepare_data('training.csv')
# pca_X_train, pca_X_test = util.pca_data(X_train, X_test, explain_threshold=0.99)

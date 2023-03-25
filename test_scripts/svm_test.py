import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from models.svm import SVM
ROOT_DIR ="
model_dir = os.path.join(ROOT_DIR, 'model_pkl')
svr_model = joblib.load(os.path.join(model_dir,"svr_model.pkl")
                        

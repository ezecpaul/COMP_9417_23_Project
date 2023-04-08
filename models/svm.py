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

class SVR_Search:
    @staticmethod
    # define base model
    def tune(X:pd.DataFrame, y:pd.DataFrame, file_name: str =None):
        
        # Load and return saved searchCV files
        if file_name:
            if os.path.exists(file_name):
                file = os.path.join(saved_searchCV_dir, file_name)                
                return joblib.load(file)  
        
        # else: Do a grid-search 
        print('\nSVR Tuning in progress... ')
        
        label = y.columns[0]
        X= np.array(X)
        y= np.array(y).ravel()
        
        # grid-search parameters range
        params = { 'kernel': ('linear', 'rbf', 'poly'),
                  'C':[1000.0, 10000.0, 15000.0],
                  'degree': [2, 3],
                  'coef0': [0.01, 0.05, 0.1]
                  }

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)       

        # initiate gridserach
        gs = GridSearchCV(estimator= svm.SVR(gamma= 'auto'), # base model
                            param_grid=params, 
                            cv=3, 
                            n_jobs=-1, 
                            verbose=1,
                            scoring=scorer )

        gs.fit(X, y)
        # Save each of searchCV as Pickle
        joblib.dump(gs, os.path.join(saved_searchCV_dir, label+'.pkl'))
        print(f'Completed and Saved as {label}.pkl in {saved_searchCV_dir}')

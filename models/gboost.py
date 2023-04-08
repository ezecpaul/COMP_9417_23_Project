import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
saved_searchCV_dir = os.path.join(ROOT_DIR, 'gbr_searchCV_pkl')

class GBR_Search:
    @staticmethod
    def tune(X:pd.DataFrame, y:pd.DataFrame, file_name: str =None):

        # Load and return saved searchCV files
        if file_name:
            if os.path.exists(file_name):
                file = os.path.join(saved_searchCV_dir, file_name)                
                return joblib.load(file)   
        
        # else: Do a grid-search and build new model
        print('\nGBR Tuning in progress... ')
        
        label = y.columns[0]
        X= np.array(X)
        y= np.array(y).ravel()

        # create an instance of GradientBoostingRegressor for each response variable and define the hyperparameters to search over
        params = {'n_estimators': [50, 100, 200], 
                  'learning_rate': [0.01, 0.1, 1.0], 
                  'max_depth': [2, 3, 5, 7] }

        scorer = make_scorer(mean_squared_error, greater_is_better=False)

        gs= GridSearchCV(estimator= GradientBoostingRegressor( loss='absolute_error', random_state=0), # loss='absolute_error' returned error
                                    param_grid=params, 
                                    cv=3, 
                                    n_jobs=-1, 
                                    verbose=1, 
                                    scoring=scorer)
        gs.fit(X, y)
        # Save Pickle
        joblib.dump(gs, os.path.join(saved_searchCV_dir, label+'.pkl'))
        print(f'Completed and Saved as {label}.pkl in {saved_searchCV_dir}')

# Running Grid-Search using all Predictors
from utils import util
data_file = 'training.csv'
X, _, y, _ = util.process_data(data_file)
for column in list(y.columns):
    y = y[[column]]
    GBR_Search.tune(X, y)
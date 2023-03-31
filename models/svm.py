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
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')
# model_file = os.path.join(saved_model_dir, 'svr_model.pkl')
svr_metrics_dir = os.path.join(saved_model_dir, 'svr_metrics')

class SVR_Model:
    @staticmethod
    # define base model
    def svm_tuning(X:pd.DataFrame, y:pd.DataFrame):
        
        # Load and return saved model if exist
        if os.path.exists(os.path.join(svr_metrics_dir, 'svr_params.csv')):
            try:
                model_best_param = pd.read_csv(os.path.join(svr_metrics_dir, 'svr_params.csv'))
                print(f'\nHere is best param for svm.SVR model \n {model_best_param}')                
                return
            except:
                print(f'\nError loading existing best parameter file')
                sys.exit(0)       
        
        # else: Do a grid-search and build new model
        print('\nSVR Tuning in progress... ')
        X = np.array(X)
        y = np.array(y)
        y = y[:,1] # build model will only one response variable (1st Response Variable)
        
        # grid-search parameters range
        param = {   'kernel': ('linear', 'rbf', 'poly'),
                    'C':[1000.0, 10000.0, 15000.0],
                    'degree': [2, 3],
                    'coef0': [0.01, 0.05, 0.1]
        }

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)       

        # define gridserach
        grid_search = GridSearchCV( estimator = svm.SVR( gamma= 'auto'), # base model declaration
                                    param_grid = param, 
                                    cv = 3, 
                                    n_jobs = -1, 
                                    verbose = 2,
                                    scoring=scorer
        )

        # tune model with grid_search
        svr_search_run = grid_search.fit(X, y)

        # save csv of grid_search results
        pd.DataFrame(svr_search_run.cv_results_).to_csv(os.path.join(svr_metrics_dir,'svr_cv_result.csv'), index=False)
        pd.DataFrame(svr_search_run.best_estimator_).to_csv(os.path.join(svr_metrics_dir,'svr_estimator.csv'), index=False)
        pd.DataFrame(svr_search_run.best_params_).to_csv(os.path.join(svr_metrics_dir,'svr_params.csv'), index=False)
        pd.DataFrame(svr_search_run.best_score_).to_csv(os.path.join(svr_metrics_dir,'svr_score.csv'), index=False)

        joblib.dump(svr_search_run.best_params_, os.path.join(processed_data_dir, 'X_train.pkl')) #save to pickle

        print(f'\n Hyperparameter tuning completed with metrics saved in {svr_metrics_dir} \n  \
                Best parameters:\n {svr_search_run.best_params_}')
        return
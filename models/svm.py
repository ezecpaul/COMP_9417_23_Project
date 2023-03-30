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
model_file = os.path.join(saved_model_dir, 'svr_model.pkl')
svr_metrics_dir =os.path.join(model_file, 'svr_metrics')


class SVR_Model:
    @staticmethod
    # define base model
    def svm_tuning(X:pd.DataFrame, y:pd.DataFrame):
        
        # Load and return saved model if exist
        if os.path.exists(os.path.join(svr_metrics_dir, 'svr_params.csv')):
            try:
                model_best = pd.read_csv(os.path.join(svr_metrics_dir, 'svr_params.csv'))
                print(f'\nHere is best param for svm.SVR model \n {model_best}')                
                return
            except:
                print(f'\nError loading existing best parameter file')
                sys.exit(0)       
        
        # else: Do a grid-search and build new model
        print('\nSVR Tuning in progress... ')
        X = np.array(X)
        y = np.array(y)
        y = y[:,0] # build model will only one response variable (1st Response Variable)
        
        # grid-search parameters range
        param = {   'kernel': ('linear', 'rbf', 'poly'),
                    'C':[1000.0, 10000.0, 15000.0],
                    'degree': [2, 3],
                    'coef0': [0.01, 0.05, 0.1],
                    'gamma': 'auto',
        }

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)       

        # define gridserach
        grid_search = GridSearchCV( estimator = svm.SVR(), # base model declaration
                                    param_grid = param, 
                                    cv = 3, 
                                    n_jobs = -1, 
                                    verbose = 2,
                                    scoring=scorer
        )

        # tune model with grid_search
        svr_search_run = grid_search.fit(X, y)

        # save csv of grid_search results
        pd.DataFrame(svr_search_run.cv_results_).to_csv(os.path.join(saved_model_dir,'svr_metrics','svr_cv_result.csv'), index=False)
        pd.DataFrame(svr_search_run.best_estimator_).to_csv(os.path.join(saved_model_dir,'svr_metrics','svr_estimator.csv'), index=False)
        pd.DataFrame(svr_search_run.best_params_).to_csv(os.path.join(saved_model_dir,'svr_metrics','svr_params.csv'), index=False)
        pd.DataFrame(svr_search_run.best_score_).to_csv(os.path.join(saved_model_dir,'svr_metrics','svr_score.csv'), index=False)

        print(f'\n Hyperparameter tuning completed with metrics saved in {svr_metrics_dir} \n  \
                Best parameters:\n {svr_search_run.best_params_}')
        return 

from utils import util

# Runing snipet #1
# process data from train dataset 'training.csv'
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = util.process_data('training.csv')

# best params = {'C': 10000.0, 'coef0':0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
model = svm.SVR(C= 10000.0, 
                coef0=0.01, 
                degree= 3, 
                gamma= 'auto', 
                kernel= 'poly')

# predict using best model
pred = util.fit_predict(model, 
                      X_train, 
                      y_train, 
                      X_test)

# Compute RMSE for each response variable and column-wise Average-RMSE
RMSEs, Ave_RMSE = util.average_RMSE(y_test, pred)
print(f'\nColumns [Ca, P, pH, SOC , Sand] RMSEs: \n {RMSEs} \n Average RMSE: {Ave_RMSE}')

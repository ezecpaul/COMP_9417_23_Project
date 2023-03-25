import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


class SVM:
    @staticmethod
    # define base model
    def svm_tuning(X:pd.Dataframe, y:pd.DataFrame) -> pd.DataFrame:
        
        # gridsearch parameter declaration
        param = {   'kernel': ('linear', 'rbf', 'poly'),
                    'C':[10, 100, 1000],
                    'degree': [2, 3],
                    'coef0': [0.01, 0.5, 1],
                    'gamma': ['auto', 0.1],
        }

        # model declaration           
        svr = svm.SVR() 

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        # hyper-parameter tuning
        grid_search = GridSearchCV( estimator = svr,
                                    param_grid = param, 
                                    cv = 3, 
                                    n_jobs = -1, 
                                    verbose = 2,
                                    scoring=scorer
                                    )

        # Fit optimised base_models
        svr_model = grid_search.fit(X, y)

        # Save model as Pickle
        joblib.dump(svr_model, "svr_model.pkl")
        return svr_model
        
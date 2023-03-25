import os
import sys
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class GBOOST:
    @staticmethod
    # define base model
    def gboost_tuning(X:pd.Dataframe, y:pd.DataFrame) -> pd.DataFrame:
        
        # gridsearch parameter declaration
        param = {   'kernel': ('linear', 'rbf', 'poly'),
                    'C':[10, 100, 1000],
                    'degree': [2, 3],
                    'coef0': [0.01, 0.5, 1],
                    'gamma': ['auto', 0.1],
        }

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        # hyper-parameter tuning
        grid_search = GridSearchCV( estimator = xgb(),
                                    param_grid = param, 
                                    cv = 3, 
                                    n_jobs = -1, 
                                    verbose = 2,
                                    scoring=scorer
                                    )

        # Fit optimised base_models
        xgb_model = grid_search.fit(X, y)

        # Save model as Pickle
        joblib.dump(xgb_model, "xgb_model.pkl")
        return xgb_model
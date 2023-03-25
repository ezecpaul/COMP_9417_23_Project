import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Define some utility functions in util class
class util:
    '''
    Define series of help functions

    ''' 
    # Compute Column-wise Average RMSE 
    @staticmethod
    def average_RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> list:
        column_len = y.shape[1] 
        errors = np.zeros(column_len)
        for i in range(column_len):
            errors[i] = mean_squared_error(y[:,i], y_hat[:,i], squared=False)
        return [errors, np.average(errors)]
    
    # Feature Variables Standardization
    @staticmethod
    def standardize(X_train: pd.DataFrame, X_test: pd.DataFrame):
        columns = X_train.columns.values
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        scaler = StandardScaler()
        scaler.fit(X_train)
        scaled_Xtrain = pd.DataFrame(data=scaler.transform(X_train), columns=columns)
        scaled_Xtest= pd.DataFrame(data=scaler.transform(X_test), columns=columns)
        return scaled_Xtrain, scaled_Xtest
    
    # Prepare and Save Submission file
    @staticmethod
    def submission(output_sample_file: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
        output = pd.read_csv(output_sample_file)
        output['Ca'] = pred[:,0]
        output['P'] = pred[:,1]
        output['pH'] = pred[:,2]
        output['SOC'] = pred[:,3]
        output['Sand'] = pred[:,4]
        output.to_csv('predictions.csv', index = False)
        print('prediction file saved')

    # Perform prediction using a built model
    @staticmethod
    def model_predict(model: joblib, X_train: pd.DataFrame, y_train: pd.DataFrame):
        n_rows = y_train.shape[0]
        n_columns = y_train.shape[1]
        y_preds = np.zeros((n_rows, n_columns))
        for i in range(n_columns):
            model.fit(X_train, y_train[:,i])
            y_preds[:,i] = model.predict(X_test).astype(float)
        return y_preds
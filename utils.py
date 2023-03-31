import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')
processed_data_dir = os.path.join(ROOT_DIR, 'data_pkl')
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')

# Define some utility functions in util class
class util:
    '''
    Define series of help functions

    ''' 
    # Compute Column-wise Average RMSE 
    @staticmethod
    def average_RMSE(y_true: pd.DataFrame, y_pred):
        columns = y_true.columns.values
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        column_len = y_true.shape[1] 
        errors = np.zeros(column_len)
        for i in range(column_len):
            errors[i] = mean_squared_error(y_true[:,i], y_pred[:,i], squared=False)
        return errors, np.average(errors)
    
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
    def submission(output_sample_file: pd.DataFrame, pred) -> pd.DataFrame:
        pred = np.array(pred)
        output = pd.read_csv(output_sample_file)

        output['Ca'] = pred[:,0]
        output['P'] = pred[:,1]
        output['pH'] = pred[:,2]
        output['SOC'] = pred[:,3]
        output['Sand'] = pred[:,4]
        output.to_csv(os.path.join(processed_data_dir, 'predictions.csv'), index = False)
        print(f'prediction file saved to {processed_data_dir}')

    # Perform prediction using a built model
    @staticmethod
    def fit_predict(model, 
                      X_train: pd.DataFrame, 
                      y_train: pd.DataFrame, 
                      X_test: pd.DataFrame):
        columns = y_train.columns.values
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)

        n_columns = y_train.shape[1]
        n_rows = X_test.shape[0]
        y_pred = np.zeros((n_rows, n_columns))        
        for i in range(n_columns):            
            y_pred[:,i]= model.fit(X_train, y_train[:,i]).predict(X_test).astype(float)
        y_pred =pd.DataFrame(data=y_pred, columns=columns)
        return y_pred

    # Process Dataset into Train and Test sets   
    @staticmethod
    def process_data(   training_data_file: str, 
                        test_data_file: str = None ):
        # check if files exist, load and return them
        if os.path.exists(os.path.join(processed_data_dir, 'X_train.csv')):
            try:
                X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'))
                X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
                y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'))
                y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'))
                X_train_scaled = pd.read_csv(os.path.join(processed_data_dir,'X_train_scaled.csv'))
                X_test_scaled = pd.read_csv(os.path.join(processed_data_dir,'X_test_scaled.csv'))

                print('\nSix(6) Dataframes (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), returned.')
                return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
            except:
                print("Error reading one of the files")            

        print('Data pre-processing in progress...\n')
        # # process test dataset if required
        # if test_data_file:
        #     test_file = os.path.join(data_dir, test_data_file)
        #     test = pd.read_csv(test_file)
        #     test.drop(['PIDN', 'Depth'], axis=1, inplace=True)

        # Read train datasets
        train_file = os.path.join(data_dir, training_data_file)
        train = pd.read_csv(train_file)
        train.dropna(inplace=True) # drop nan row
        labels = train[['Ca','P','pH','SOC','Sand']] # y/labels

        # drop response values(labels), ID, and categorical variable from predictors
        train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN','Depth'], axis=1, inplace=True)        

        # split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.20, random_state=100) 
        
        # reset indexes
        X_train.reset_index(drop=True, inplace=True) 
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # save csv files
        X_train.to_csv(os.path.join(processed_data_dir,'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_data_dir,'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(processed_data_dir,'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(processed_data_dir,'y_test.csv'), index=False)
        # joblib.dump(X_train, os.path.join(processed_data_dir, 'X_train.pkl')) #save to pickle

        # 1st: scale and save tranformed data: util.standardize(X_train, X_test)
        X_train_scaled, X_test_scaled = util.standardize(X_train, X_test)

        X_train_scaled.to_csv(os.path.join(processed_data_dir,'X_train_scaled.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(processed_data_dir,'X_test_scaled.csv'), index=False)

        # 2nd: PCA- Feature Reduction and Selection

        # 3nd: Pearson Correlation for Feature Reduction and Selection

        # 4th: Spearman Correlation for Feature Reduction and Selection

        # 5th VIF Feature Reduction and Selection
    

        print('\nSix(6) Dataframes (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), returned.')
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test

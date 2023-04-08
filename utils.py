import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error
import joblib
from feature_select import FS
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
    def standardize(X_train: pd.DataFrame, X_test: pd.DataFrame, normalize=False):
        columns = X_train.columns.values
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Standardize X to have zero mean and unit variance 
        # Note: data have been mean centered and scaled
        scaler = StandardScaler()
        scaled_Xtrain = scaler.fit_transform(X_train)
        scaled_Xtest = scaler.transform(X_test)

        if not normalize:
            return scaled_Xtrain, scaled_Xtest

        # Normalize/transform each sample to have unit norm
        
        normalizer = Normalizer()
        normalized_X_train = normalizer.fit_transform(scaled_Xtrain)

        # Normalize the test data using the same normalization parameters as the training data
        normalized_X_test = normalizer.transform(scaled_Xtest)

        Xtrain= pd.DataFrame(data= normalized_X_train, columns=columns)
        Xtest= pd.DataFrame(data= normalized_X_test, columns=columns)        
        return Xtrain, Xtest
    
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

    # Perform model fit
    @staticmethod
    def model_fit(model, X_train: pd.DataFrame, y_train: pd.DataFrame):
        columns = y_train.columns.values
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        models =[]
        n_rows = y_train.shape[0]
        n_columns = y_train.shape[1]     
        for i in range(n_columns):            
            model.fit(X_train, y_train[:,i])
            models.append(model)
        return models

    # Model predict    
    @staticmethod
    def model_predict(models, X_test: pd.DataFrame):
        X_test = np.array(X_test)
        y_pred = np.array((X_test.shape[0], len(models)))
        for i in range(len(models)):
           y_pred[:,i] = models[i].predict(X_test).astype(float)
        return y_pred

    # Process Dataset into Train and Test sets   
    @staticmethod
    def prepare_data(training_data_file: str, test_data_file: str = None):
                                  
        print('Data preparation in progress...\n')
        # # process test dataset if required

        
        # Read train datasets
        train_file = os.path.join(data_dir, training_data_file)
        train = pd.read_csv(train_file)
        train.dropna(inplace=True) # drop nan row
        labels = train[['Ca','P','pH','SOC','Sand']] # y/labels

        # drop response values(labels), ID, and categorical variable from predictors
        train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN','Depth'], axis=1, inplace=True)        

        # split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.20, random_state=100) 
        
        if test_data_file:
            test_file = os.path.join(data_dir, test_data_file)
            test = pd.read_csv(test_file)
            test.drop(['PIDN', 'Depth'], axis=1, inplace=True)

            X_train= pd.concat([X_train, X_test], ignore_index=True)
            y_train = pd.concat([y_train, y_test], ignore_index=True)
            X_test = test
            y_test = np.zeros(test.shape)
            

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
        
        print(f'Completed! - Files saved in {processed_data_dir}')
        return X_train, X_test, y_train, y_test

    def pca_data(X_train: pd.DataFrame, X_test: pd.DataFrame, explain_threshold:float ):
        print('PCA Data Preparation in progress...\n')
        pca, top_features = FS.pca(X_train, explain_threshold=explain_threshold)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        df_train = pd.DataFrame(X_train_pca, columns=['PC-'+str(i) for i in range(1, X_train_pca.shape[1]+1)], 
                        index=[i for i in range(X_train_pca.shape[0])])
        df_test = pd.DataFrame(X_test_pca, columns=['PC-'+str(i) for i in range(1, X_test_pca.shape[1]+1)])

        df_train.to_csv(os.path.join(processed_data_dir,'X_train_pca.csv'), index=False)
        df_test.to_csv(os.path.join(processed_data_dir,'X_test_pca.csv'), index=False)

        print(f'Completed! - Files saved in {processed_data_dir}')
        return df_train, df_test



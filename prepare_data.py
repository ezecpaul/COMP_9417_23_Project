import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')
processed_data_dir = os.path.join(ROOT_DIR, 'data_pkl')
saved_model_dir = os.path.join(ROOT_DIR, 'model_pkl')



def prepare_data(train_data: pd.DataFrame, test_split_ratio = 0.2):
                                  
        print('Data preparation in progress...')

        labels = train_data[['Ca','P','pH','SOC','Sand']] # y/labels

        # drop response values(labels) and categorical variable from predictors
        train_data.drop(['Ca', 'P', 'pH', 'SOC', 'Sand'], axis=1, inplace=True)        

        # split into train/test sets
        if test_split_ratio > 0.0:
            X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=test_split_ratio, random_state=100) 
        
            # reset indexes
            X_train.reset_index(drop=True, inplace=True) 
            X_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)

        else:
            X_train = train_data
            y_train = labels
            X_test = None
            y_test = None

        # save csv files
        X_train.to_csv(os.path.join(processed_data_dir,'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(processed_data_dir,'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(processed_data_dir,'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(processed_data_dir,'y_test.csv'), index=False)     
        print(f'Completed! - Files saved in {processed_data_dir}')

        print('Completed!')
        return X_train, X_test, y_train, y_test

def pca_data(X_train: pd.DataFrame, X_test: pd.DataFrame, explain_threshold: float ):
    print('PCA Data Preparation in progress...\n')
    from feature_select import FS
    pca, top_features = FS.pca(X_train, explain_threshold=explain_threshold)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    df_train = pd.DataFrame(X_train_pca, columns=['PC-'+str(i) for i in range(1, X_train_pca.shape[1]+1)], 
                    index=[i for i in range(X_train_pca.shape[0])])
    df_test = pd.DataFrame(X_test_pca, columns=['PC-'+str(i) for i in range(1, X_test_pca.shape[1]+1)])

    df_train.to_csv(os.path.join(processed_data_dir,'X_train_pca.csv'), index=False)
    df_test.to_csv(os.path.join(processed_data_dir,'X_test_pca.csv'), index=False)
    print(f'Completed! - Files saved in {processed_data_dir}')

    print('Done!')
    return df_train, df_test

def clean_outliers(df : pd.DataFrame, drop_outliers = False )-> pd.DataFrame:
    
    print(f'\nCleaning data {"with outliers dropped" if drop_outliers else "with outliers replaced"}')
    
    df.drop(['PIDN', 'Depth'], axis=1, inplace=True)
    y = df[['Ca','P','pH','SOC','Sand']]
    
    # remove outliers
    # calculate the interquartile range for each response feature
    q3  = np.percentile(y, 75, axis=0)
    q1 = np.percentile(y, 25, axis=0)
    iqr = q3 - q1

    # define a threshold for outlier detection
    outlier_thresh = 1.5

    lower_end = q1 - outlier_thresh*iqr
    upper_end = q3 + outlier_thresh*iqr

    if drop_outliers:
        
        # Drop outliers = True
        outliers = (y < (lower_end)) | (y > (upper_end))
        df = df[~outliers.any(axis=1)]
    
    else:
        
        # Drop outliers = False
        for i in range(5):
            df[y.columns[i]] = np.where(df[y.columns[i]] < lower_end[i], lower_end[i], df[y.columns[i]])
            df[y.columns[i]] = np.where(df[y.columns[i]] > upper_end[i], upper_end[i], df[y.columns[i]])

    
    df.reset_index(drop=True, inplace=True)

    # replace null value with column mean
    df.fillna(df.mean(), inplace=True)

    print('Done!\n')
    # return cleaned dataframe
    return df

# Running Grid-Search using all Predictors
if __name__ == "__main__":

    # Read in train Dataset 
    file = os.path.join(data_dir, 'training.csv')
    df =  pd.read_csv(file)

    #### 1: Data Cleaning and Normarlization
    
    #import utility/helper function class
   
    # clean data and drop or replace outliers- set drop_outlers to True to drop or False (default) to replace them
    data = clean_outliers(df, drop_outliers = True )

    # set test_split_ratio =0.0 ( X_test and y_test becomes None) when preparing for Final Submission Otherwise set to (0.1 or 0.2): default-0.2
    X_train, X_test, y_train, y_test = prepare_data(data, test_split_ratio=0.2)


    # 2: Feature Reduction/Selection

    # >>> 2.1 PCA <<<
    # import Feature Transformation class FS from file feature_select.py
    from feature_select import FS
    pca, top_features = FS.pca(X_train, explain_threshold=0.99)

    # transform the pca tranform X_train and X_test in ndarry pca versions
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
import os
import sys
import pandas as pd
import numpy as np
import joblib
sys.path.append('.')

def process_data():
    print('\t Data Pre-Processing in Progress...\n')
    ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory

    # define dataset files
    data_dir = os.path.join(ROOT_DIR, 'data')
    processed_data_dir = os.path.join(ROOT_DIR, 'data_pkl')

    train_file = os.path.join(data_dir, 'training.csv')
    test_file = os.path.join(data_dir, 'sorted_test.csv')

    # Read train & test datasets
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    labels = train[['Ca','P','pH','SOC','Sand']] # y/labels

    # train.isnull().values.any() # checked if any cell contains null value
    # train = np.array(train)[:,:3578] # convert train set to numpy array

    # drop response values(labels), ID, and categorical variable from predictors
    train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN','Depth'], axis=1, inplace=True)
    test.drop(['PIDN', 'Depth'], axis=1, inplace=True)

    # split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.20, random_state=100) 

    # reset indexes
    X_train = X_train.reset_index(drop=True) 
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # save csv files
    X_train.to_csv(os.path.join(processed_data_dir,'X_train.csv'))
    X_test.to_csv(os.path.join(processed_data_dir,'X_test.csv'))
    y_train.to_csv(os.path.join(processed_data_dir,'y_train.csv'))
    y_test.to_csv(os.path.join(processed_data_dir,'y_test.csv'))

    # joblib.dump(X_train, os.path.join(processed_data_dir, 'X_train.pkl')) #save to pickle

    print('\t Completed with csv of "X_train, X_test, y_train and y_test" saved in data_pkl directory')
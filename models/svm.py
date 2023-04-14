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
saved_searchCV_dir = os.path.join(ROOT_DIR, 'svr_searchCV_pkl')
data_dir = os.path.join(ROOT_DIR, 'data')

class SVR_Search:
    @staticmethod
    # define base model
    def tune(X:pd.DataFrame, y:pd.DataFrame, file_name: str =None):
        
        # Load and return saved searchCV files
        if file_name:
            if os.path.exists(file_name):
                file = os.path.join(saved_searchCV_dir, file_name)                
                return joblib.load(file)  
        
        # else: Do a grid-search 
        print('\nSVR Tuning in progress... ')
        
        label = y.columns[0]
        X= np.array(X)
        y= np.array(y).ravel()
        
        # grid-search parameters range
        params = { 'kernel': ('linear', 'rbf', 'poly'),
                  'C':[1000.0, 10000.0, 15000.0],
                  'degree': [2, 3],
                  'coef0': [0.01, 0.05, 0.1]
                  }

        # scorer
        scorer = make_scorer(mean_squared_error, greater_is_better=False)       

        # initiate gridserach
        gs = GridSearchCV(estimator= svm.SVR(gamma= 'auto'), # base model
                            param_grid=params, 
                            cv=3, 
                            n_jobs=-1, 
                            verbose=1,
                            scoring=scorer )

        gs.fit(X, y)
        # Save each of searchCV as Pickle
        joblib.dump(gs, os.path.join(saved_searchCV_dir, label+'.pkl'))
        print(f'Completed and Saved as {label}.pkl in {saved_searchCV_dir}')

# Running Grid-Search using all Predictors
if __name__ == "__main__":

    # Read in train Dataset 
    file = os.path.join(data_dir, 'training.csv')
    df =  pd.read_csv(file)

    #### 1: Data Cleaning and Normarlization
    
    #import utility/helper function class
    from utils import util

    # clean data and drop or replace outliers- set drop_outlers to True to drop or False (default) to replace them
    data = util.clean_outliers(df, drop_outliers = True )

    # set test_split_ratio =0.0 ( X_test and y_test becomes None) when preparing for Final Submission Otherwise set to (0.1 or 0.2): default-0.2
    X_train, X_test, y_train, y_test = util.prepare_data(data, test_split_ratio=0.2)


    # 2: Feature Reduction/Selection

    # >>> 2.1 PCA <<<
    # import Feature Transformation class FS from file feature_select.py
    from feature_select import FS
    pca, top_features = FS.pca(X_train, explain_threshold=0.99)

    # transform the pca tranform X_train and X_test in ndarry pca versions
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    #>>> 2.2 Correlation Analysis <<<
    # get filtered pearson and spearman corr feature sets for all response variable with >= 0.1 corr coef
    # pearson = FS.pearson_corr_filtered(X_train, y_train, 0.1)
    spearman = FS.spearman_corr_filtered(X_train, y_train, 0.1)

    # 3: Modeling

    # 3.1 Hyperparameter Tuning
    # Grid search completed on with all feature space used and the pickle files imported
    # import pickle files of Grid-search and keep in a list
    # import pickle files and keep in a list
    svr_dir = os.path.join(ROOT_DIR, 'svr_searchCV_pkl') # directory where GridsearchCV files are stored
    files = ['Ca.pkl', 'P.pkl', 'pH.pkl', 'Sand.pkl', 'Soc.pkl']
    searchCVs = list()
    for i in range(5):
        cv_file = joblib.load(os.path.join(svr_dir, files[i]))
        searchCVs.append(cv_file)

    # 3.2 Models: Best parameters applied to build models + Model fit
    
    ## Choose  data set to fit model on
    print('What data to fit model on ')
    print('Enter: 1: Original Feature Sets')
    print('Enter: 2: PCA Reduced Feature Sets')
    print('Enter: 3: Spearmans Corr Reduced Feature Sets')
    # ans = input('Enter 1, 2, or 3:  ')
    flag = True
    while(flag):
        ans = int(input('Enter 1, 2, or 3:  '))
        if ans not in (1, 2, 3):
            print('Wrong value Entered')
            flag = True
        else:
            flag = False

    if ans == 1:
    # **On Original Data**
        print('\nfitting model on original feature space...\n')
        labels = list(y_train.columns)
        models =list()
        y_hat = np.zeros(y_train.shape)
        y_pred = np.zeros(y_test.shape)
        for i in range(5):
            X, y = np.array(X_train), np.array(y_train[labels[i]])
            model = svm.SVR(**searchCVs[i].best_params_).fit(X, y)
            y_hat[:, i] = model.predict(X)
            y_pred[:, i] = model.predict(X_test)
            models.append(model)
    
        # Report train and test errors
        train_errors, mcrmse_train = util.MCRMSE(np.array(y_train), y_hat )
        test_errors, mcrmse_test = util.MCRMSE(np.array(y_test), y_pred )
        print(f'Train errors:{train_errors} | Train MCRMSE:{mcrmse_train}')
        print(f'Test errors: {test_errors} | Test MCRMSE: {mcrmse_test}')
            
    elif ans == 2:
        print('\nfitting model on PCA reduced feature space...\n')
        # **PCA transformed Data**
        y_hat2 = np.zeros(y_train.shape)
        y_pred2 = np.zeros(y_test.shape)
        for i in range(5):
            X, y = np.array(X_train_pca), np.array(y_train[labels[i]])
            model = svm.SVR(**searchCVs[i].best_params_).fit(X, y)
            y_hat2[:, i] = model.predict(X)
            y_pred2[:, i] = model.predict(X_test_pca)

        # train and test errors
        train_errors, mcrmse_train = util.MCRMSE(np.array(y_train), y_hat2 )
        test_errors, mcrmse_test = util.MCRMSE(np.array(y_test), y_pred2 )
        print(f'Train errors:{train_errors} | Train MCRMSE:{mcrmse_train}')
        print(f'Test errors: {test_errors} | Test MCRMSE: {mcrmse_test}')
    
    elif ans == 3:
        print('\nfitting model on spearman corr reduced feature space...\n')
        y_hat4 = np.zeros(y_train.shape)
        y_pred4 = np.zeros(y_test.shape)

        for i, k in enumerate(pearson.keys()):
            feature_list = list(spearman[k].index)
            X, y = X_train[feature_list], np.array(y_train[labels[i]])
            X_ = X_test[feature_list]
            model = svm.SVR(**searchCVs[i].best_params_).fit(X, y)
            y_hat4[:, i] = model.predict(X)
            y_pred4[:, i] = model.predict(X_)

        # train and test errors
        train_errors, mcrmse_train = util.MCRMSE(np.array(y_train), y_hat4 )
        test_errors, mcrmse_test = util.MCRMSE(np.array(y_test), y_pred4 )
        print(f'Train errors:{train_errors} | Train MCRMSE:{mcrmse_train}')
        print(f'Test errors: {test_errors} | Test MCRMSE: {mcrmse_test}')

    ## end
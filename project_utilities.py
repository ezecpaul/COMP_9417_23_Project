import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Utilities:
    '''
    Define series of help functions

    ''' 
    @staticmethod
    def average_RMSE(y, y_hat): # Define column-wise average RMSE
        column_len = y.shape[1] 
        errors = np.zeros(column_len)
        for i in range(column_len):
            errors[i] = mean_squared_error(y[:,i], y_hat[:,i], squared=False)
        return errors, np.average(errors)
    
    @staticmethod
    def normalizer(Xtrain, Xtest, col):
        scaler = StandardScaler()
        scaler.fit(Xtrain)
        scaled_Xtrain = pd.DataFrame(data=scaler.transform(Xtrain), columns=col)
        scaled_Xtest= pd.DataFrame(data=scaler.transform(Xtest), columns=col)
        return scaled_Xtrain, scaled_Xtest
   
    @staticmethod
    def pca(df: pd.DataFrame, explain_thres: float, mode: str):
        """
        list of numeric features based on variance the component explaind
        :param df: Dataframe of feature variables
        :param explain_thres: variance threshold to required. 
        :param mode (criteria)-
              "total" : overall varience explained. 
               "ind": variance of each component explains.
        :return: PCA class object
        """ 
        if mode == "ind":
            # PCA - Get all components first
            pca_test = PCA(random_state = 42)
            pca_test.fit(df)

            # Get number of components exceeding threshold
            n_comp = sum(pca_test.explained_variance_ratio_ > explain_thres)
            print(f'There are {n_comp} components with variance explained > {explain_thres}')
        
            # Run PCA with the number of components such that variance explained for each component exceeds threshold
            pca_fin = PCA(n_components= n_comp, random_state = 42)
            pca_fin.fit(df)
        else:
            pca_fin = PCA(n_components= explain_thres, svd_solver = 'full', random_state = 42)
            pca_fin.fit(df)
            n_comp = len(pca_fin.explained_variance_ratio_)
            print(f'There are {n_comp} components with total variance explained > {explain_thres}')
        
        print(f'Total variance explained: {sum(pca_fin.explained_variance_ratio_):.4f}')

        return pca_fin
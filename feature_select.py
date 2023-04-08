import time
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import Lasso

class FS:

    @staticmethod
    def pearson_corr(X: pd.DataFrame, y: pd.DataFrame, n_features) -> dict():
        """
        Returns a dictionary of top [n_features and their Correlation] for each response variable as the key
        NOTE: Param: n_feature should be any integer less than total number of features
        """
        # assert n_features less than total features
        if n_features > X.shape[1]:
            print(f'\n{n_features} is greater {len(X.shape[1])} features')
            sys.exit(0)
            return

        print(f'\nRuning Pearson correlation for {n_features} important feartures for each target...\n')

        # concat X, y - Because of the relationship amongst response variables
        df = pd.concat([X, y], axis=1)

        # timer starts
        tic = time.perf_counter()

        # Dict container of each lablel and Pandas series of features with top correlation
        top_features_list = {} 

        for i in range(y.columns.size):
            label = y.columns[i]
            # Get correlations btw predictors and each response Variable
            corr_matrix = df.corr(method='pearson')[label].abs().sort_values(ascending=False)
            top_features = corr_matrix.iloc[:n_features + 1]  # plus 1 for target variable
            top_features.drop(label, inplace=True)
            top_features_list[label] = top_features
        
        # timer ends
        toc = time.perf_counter()
        print(f"Performance: {toc - tic:0.2f} seconds")
        return top_features_list


    def pearson_corr_filtered(X : pd.DataFrame, y: pd.DataFrame, threshold: float)  -> dict():
        """
        Returns a dictionary of top [n_features and their Correlation] for each response variable as the key

        NOTE:
        threshold could be set between 0.0 and 1.0 
        Cohen's: >0.5 = is strong, >0.3 = is moderate, >0.1 = is weak
        Recommended threshold: >0.1. Notable, the 'P' response variable has shown a lot of weak correlation to predictors)
        """
        print(f'\nRuning Pearson correlation for important features with correlations > {threshold} ...\n')

        # concat X, y - Because of the relationship amongst response variables
        df = pd.concat([X, y], axis=1)

        # timer starts
        tic = time.perf_counter()

        # Dict container of each lablel and Pandas series of features with top correlation
        top_features_list = {} 

        for i in range(y.columns.size):
            # add each target column for correlation matrix
            label = y.columns[i]
            corr_matrix = df.corr(method='pearson')[label].abs().sort_values(ascending=False)
            top_features = pd.Series(corr_matrix).where(lambda x: x >= threshold).dropna()
            top_features.drop(label, inplace=True)
            top_features_list[label] = top_features
            
            print(f"{top_features.size} features selected for target: {label}")

        # timer ends
        toc = time.perf_counter()
        print(f"Performance: {toc - tic:0.2f} seconds")
        return top_features_list

    @staticmethod
    def spearman_corr(X: pd.DataFrame, y: pd.DataFrame, n_features: int) -> dict():
        """
        Returns a dictionary of top [n_features and their Correlation] for each response variable as the key
        NOTE: Param: n_feature should be any integer less than total number of features
        """

        if n_features > X.shape[1]:
            print(f'\n{n_features} is greater {len(X.shape[1])} features')
            sys.exit(0)
            return

        print(f'\nRuning Spearman correlation for {n_features} important feartures for each target...\n')

        # concat X, y - Because of the relationship among response variables
        df = pd.concat([X, y], axis=1)

        # timer starts
        tic = time.perf_counter()

        # Dict container of each lablel and Pandas series of features with top correlation
        top_features_list = {} 

        for i in range(y.columns.size):
            label = y.columns[i]
            # Get correlations btw predictors and each response Variable
            corr_matrix = df.corr(method='spearman')[label].abs().sort_values(ascending=False)
            top_features = corr_matrix.iloc[:n_features + 1]  # plus 1 for target variable
            top_features.drop(label, inplace=True)
            top_features_list[label] = top_features
        
        # timer ends
        toc = time.perf_counter()
        print(f"Performance: {toc - tic:0.2f} seconds")
        return top_features_list


    @staticmethod
    def spearman_corr_filtered(X: pd.DataFrame, y: pd.DataFrame, threshold: float) -> dict():
        """
        Returns a dictionary of top [n_features and their Correlation] for each response variable as the key

        NOTE:
        threshold could be set between 0.0 and 1.0 
        Cohen's: >0.5 = is strong, >0.3 = is moderate, >0.1 = is weak
        Recommended threshold: >0.1. Notable, the 'P' response variable has shown a lot of weak correlation to predictors)
        """

        print(f'\nRuning Pearson correlation for important features above {threshold} threshold...\n')

        # concat X, y - Because of the relationship amongst response variables
        df = pd.concat([X, y], axis=1)

        # timer starts
        tic = time.perf_counter()

        # Dict container of each lablel and Pandas series of features with top correlation
        top_features_list = {} 

        for i in range(y.columns.size):

            # add each target column for correlation matrix
            label = y.columns[i]
            corr_matrix = df.corr(method='spearman')[label].abs().sort_values(ascending=False)
            top_features = pd.Series(corr_matrix).where(lambda x: x >= threshold).dropna()
            top_features.drop(label, inplace=True)
            top_features_list[label] = top_features
            
            print(f"{top_features.size} features selected for target: {label}")

        # timer ends
        toc = time.perf_counter()
        print(f"Performance: {toc - tic:0.2f} seconds")
        return top_features_list
    

    @staticmethod
    def pca(df: pd.DataFrame, explain_threshold: float):
        pca = PCA()
        pca.fit(df)

        # Calculate the explained variance for each component
        explained_variance = pca.explained_variance_ratio_

        # Calculate the cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance)

        # Determine the number of components required to explain percentage(%) of the variability
        n_components = np.argmax(cumulative_variance >= explain_threshold) + 1

        # Fit PCA with the selected number of components
        pca = PCA(n_components=n_components)
        pca.fit(df)
    
        # Select the top features that explain the variability in the data
        top_features = np.abs(pca.components_).argsort()[:, ::-1][:, :n_components].flatten()
        return pca, top_features
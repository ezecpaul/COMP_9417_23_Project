from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import json
import os
import sys
sys.path.append('.') # point to the root directory
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')

class ELASTICNET:
    def elasticnet_tuning(self, X, y, model_path="elastic_net.pkl"):
        print("Hyperparameter tuning in progress...")
        test_size = 0.2
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=100)
        parameters = {
            'alpha': [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 5, 10],
            'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            'max_iter': [2000, 5000, 7000]
        }
        clf = ElasticNet()
        model = GridSearchCV(clf,  parameters)
        model.fit(X_train, y_train)
        best_clf = model.best_estimator_
        print("Best train score:", model.best_score_)
        with open(model_path, 'w') as f:
            json.dump(model.best_params_, f)
        
        return model

    def elasticnet_predict(self, model_path="elastic_net.pkl"):
        with open(model_path, 'r') as f:
            parameter = json.load(f)
        clf = ElasticNet(
            alpha=parameter['alpha'], max_iter=parameter['max_iter'], l1_ratio=parameter['l1_ratio'])
        return clf


if __name__ == "__main__":

    # read train dataset
    file = os.path.join(data_dir, 'training.csv')
    data_file =  pd.read_csv(file)

    #import utility/helper function class
    from utils import util
    # clean data and drop or replace outliers- set drop_outlers to True to drop or False (default) to replace them
    data = util.clean_outliers(data_file, drop_outliers = True )

    # specify X, y
    X, y = data.iloc[:, :3593], data.iloc[:, 3593:]
        
    if os.path.exists("elastic_net.pkl"):
        model = ELASTICNET().elasticnet_predict(model_path="elastic_net.pkl")
    else:
        model = ELASTICNET().elasticnet_tuning(X, y, "elastic_net.pkl")
    
    # Split Data and Train and 
    X_train, X_test, y_train, y_test = util.prepare_data(data, test_split_ratio=0.2)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("test score:", mean_squared_error(y_test, pred , squared=False) )

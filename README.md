# COMP_9417_23_Project
 UNSW COMP9417 23T1 GROUP PROJECT
 By Tunners GO

## Download Data from compettition website and save in data dir
https://www.kaggle.com/competitions/afsis-soil-properties/overview

-**data/training.csv** train set
-**data/sorted_test.csv** test set
-**data/sample_subimission.csv** Empty submisssion_sample csv 

## pre-processed dataset dir
-**data_pkl**

## Utility files
-**utils.py**
-**feature_select.py**

## Gridsearch pickle files Dir
-**svr_searchCV_pkl**
-**gbr_searchCV_pkl** 
-some tranining done in another computer architecture might pose some problems
## Run to prepare data
-**prepare_data.py** This will prepare data for training and save it in data_pkl

## To run Models
- **models/svm.py.py**: Fits Support Vector regressor and evaluation on test data
- **models/elastic_net.py**:  Fits Elastic_net regressor and evalution on test data 
- **models/rforest.py**: Fits Random Forest regressor and evalution on test data 
- **models/nnr.py**: Fits Neural Network regressor and evalution on test data 
- **models/gboost.py**: Fits Grediant Boosting regressor and evalution on test data 

## To prepare kaggle submission
-**kaggle_submission.py** product prediction on sorted_test.csv file in data directory

## kaggle submisssion csv
-**data_pkl/prediction.csv**
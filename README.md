# Customer Churn Experiment Tracking and Deployment
I got the dataset from kaggle and it is a binary clasification project about customer churn. you can see the data here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## The Project runs two containers:
* mlfow server - just hosts mlflow sqlite server on localhost 5000
* mlfow client - container for running hyperparemeter search, training single model and predicting

## mlfow_client
* rs.py
takes two optinal command line keyword arguments -d - data to optimize the model on (data.csv as default) -n number of trials for the 5 fold corss validation hyperparameter tunning. I'm using optima to optimze a lightgbm model. For each iteration and each cross validation step I'm logging accuracy and auc into mlflow (churn_random_search_cv experimet). after the trials are done a model is trained on the best hyperparameters, model accuracy and auc is evaluated on the train, validation and test sets, this gets logged into mlflow, including the model file (churn_best experiment) and the model booster object is saved as a txt into a models subdirectory.
* one.py
* predict.py

## docker-compose

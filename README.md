# Customer Churn Experiment Tracking and Deployment
I got the dataset from kaggle and it is a binary clasification project about customer churn. you can see the data here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## The Project runs two containers:
* mlfow server - just hosts mlflow sqlite server on localhost 5000
* mlfow client - container for running hyperparemeter search, training single model and predicting

## mlfow_client
* rs.py
  
  takes two optinal command line keyword arguments -d - data to optimize the model on (data.csv as default) -n number of trials for the 5 fold corss validation hyperparameter tunning. I'm using       optima to optimze a lightgbm model. For each iteration and each cross validation step I'm logging accuracy and auc into mlflow (churn_random_search_cv experimet). after the trials are done a     
  model is trained on the best hyperparameters, model accuracy and auc is evaluated on the train, validation and test sets, this gets logged into mlflow, including the model file (churn_best 
  experiment) and the model booster object is saved as a txt into a models directory. It also saves graphs that show model performance (only on the test set).

  the module gets hyperparameters to optimize from the config_rs.json. the file should have the following structure:
  * constant hyperparameter (not optimized) - "hp_name": "hp_value",
  * hyperparameter chosen from a list - "hp_name": {"type": "categorical", "value": [v1, v2, v3 ...]}
  * hyperparameter chosen from a range (int or float) - "hp_name": {"type": "int" or "float", "value": [min_value, max_value]}
  
* one.py
* predict.py

## docker-compose

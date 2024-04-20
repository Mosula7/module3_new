# Customer Churn Experiment Tracking and Deployment
I got the dataset from kaggle and it is a binary clasification project about customer churn. you can see the data here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## The Project runs two containers:
* mlflow server - just hosts mlflow sqlite server on localhost 5000
* mlflow client - container for running hyperparemeter search, training single model and predicting
* to run both containers by:
```
docker-compose up
```

## mlflow_client
* rs.py
  
  takes two optinal command line keyword arguments -d - data to optimize the model on (data.csv as default) -n number of trials for the 5 fold corss validation hyperparameter tunning. I'm using       optima to optimze a lightgbm model. For each iteration and each cross validation step I'm logging accuracy and auc into mlflow (churn_random_search_cv experimet). after the trials are done a     
  model is trained on the best hyperparameters, model accuracy and auc is evaluated on the train, validation and test sets, this gets logged into mlflow, including the model file (churn_best 
  experiment) and the model booster object is saved as a txt into a models directory. It also saves graphs that show model performance (only on the test set) in the graphs directory.

  the module gets hyperparameters to optimize from the config_rs.json. the file should have the following structure:
  * constant hyperparameter (not optimized) - "hp_name": "hp_value",
  * hyperparameter chosen from a list - "hp_name": {"type": "categorical", "value": [v1, v2, v3 ...]}
  * hyperparameter chosen from a range - "hp_name": {"type": "int" or "float", "value": [min_value, max_value]}
 
  * to run the experiments in the container run
  ```
  python rs.py
  ```

* one.py

  takes one optinal command line keyword argument -d - data to train the model on (data.csv as default). it splits the data into train validation and test sets and logs auc and accuracy on these 
  three sets into mlflow server including the model file (churn_one experiment), it also saves the model booster object into models directory and graph of the models performace on the test set in 
  the graphs direcory. it takes hyperparameters from the config_one.json. in the config file write {"hp1_name": "hp1_value", "hp2_name": "hp2_value"...}.
  
  * to run the experiments in the container simply run
  ```
  python one.py
  ```
  
* predict.py

  takes the model and data name from the config_predict.json. imports data, processes it splits it into X and y, makes predictions, saves predictions in the predictions directory and saves the 
  graph of the models performance on this set in the graphs directory.

  * to run the experiments in the container simply run
  ```
  python predict.py
  ```
* helper_functions.py

  functions for data processing, data splitting, making graphs for model performance and model training and logging

* mlflow.db

  local database to track models

## docker-compose
*  creates model_server and model_client containers
*  volumes - all three config files, data, models, graphs, data and predictions directories and the database mlflow.db 


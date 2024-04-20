import json
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import optuna
from optuna.samplers import TPESampler
from functools import partial

from datetime import datetime
import os
import argparse

import mlflow
mlflow.set_tracking_uri("sqlite://../mlflow_server/mlflow.db")

with open('config_rs.json') as file:
    config = json.load(file)

  

def objective_lgbm(trial, X, y):   
    mlflow.set_experiment('churn_random_search_cv')
    experiment = mlflow.get_experiment_by_name('churn_random_search_cv')
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    with mlflow.start_run(run_id=run.info.run_id):
        # hyperparameters for the model
        params = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if value["type"] in ("int", "float"):
                    params[key] = trial.__getattribute__(f"suggest_{value["type"]}")(key, value["values"][0], value["values"][0])
                if value['type'] == "categorical":
                    params[key] = trial.__getattribute__(f"suggest_{value["type"]}")(key, value["values"])
            else:
                params[key] = value

        mlflow.log_params(params)
        # arrays for metrics
        auc_array = np.array([])
        acc_array = np.array([])

        # doing 5 fold stratified cross validation
        skf = StratifiedKFold(n_splits=5)
        for i, (train_index, valid_index) in enumerate(skf.split(X=X, y=y)):

            # splitting data
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]

            X_valid = X.iloc[valid_index]
            y_valid = y.iloc[valid_index]
            
            # initializing, fitting and predicting on validation set
            lgbm = lgb.LGBMClassifier(**params)
            lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            pred_valid = lgbm.predict_proba(X_valid)[:,-1]

            # calculating auc
            auc = roc_auc_score(y_valid, pred_valid)
            auc_array = np.append(auc_array, auc)
            mlflow.log_metric(f"val_auc_{i}", auc)

            # calculating accuracy
            acc = accuracy_score(y_valid, pred_valid>.5)
            acc_array = np.append(acc_array, acc)
            mlflow.log_metric(f"val_acc_{i}", acc)
        
        mlflow.log_metric('val_auc_avg', np.mean(auc_array))
        mlflow.log_metric('val_acc_avg', np.mean(acc_array))
        
        return np.mean(auc_array)



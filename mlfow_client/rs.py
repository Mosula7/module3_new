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
from helper_functions import process_data, train_and_log_performance

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

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
                    params[key] = trial.__getattribute__(f"suggest_{value["type"]}")(key, value["values"][0], value["values"][1])
                if value['type'] == "categorical":
                    params[key] = trial.__getattribute__(f"suggest_{value["type"]}")(key, value["values"])
            else:
                params[key] = trial.suggest_categorical(key, [value])

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


def get_best_params(target, path, n_trials):

    df = pd.read_csv(path)
    df = process_data(df)

    train, test = train_test_split(df, test_size=0.15, stratify=df[target], random_state=17)

    X = train.drop(columns=[target])
    y = train[target]
    
    objective = partial(objective_lgbm, X=X, y=y)
    sampler = TPESampler(seed=17)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, train, test



def train_best_model(n_trials, data_name):
    target = 'Churn_Yes'
    path = os.path.join('data', data_name)
    
    params, train, test = get_best_params(target, path, n_trials)

    train, val = train_test_split(train, test_size=0.15, stratify=train[target], random_state=17)

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    mlflow.set_experiment("churn_best")
    model_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    train_and_log_performance(model_name, params, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
 
    parser.add_argument("-n", "--n_trials", help = "number of trials in the hyperparameter random search", type=int, default=10)
    parser.add_argument("-d", "--data_name", help = "name of the data file", default='data.csv')

    args = parser.parse_args()

    n_trials = args.n_trials
    data_name = args.data_name

    train_best_model(n_trials, data_name)



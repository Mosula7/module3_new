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
from process_data import process_data

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



def train_best_model():
    
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-n", "--n_trials", help = "number of trials in the hyperparameter random search", type=int, default=10)
    parser.add_argument("-d", "--data_name", help = "name of the data file", default='data.csv')

    args = parser.parse_args()

    n_trials = args.n_trials
    data_name = args.data_name

    target = 'Churn_Yes'
    path = os.path.join('data', data_name)
    
    params, train, test = get_best_params(target, path, n_trials)
    print(params)

    train, val = train_test_split(train, test_size=0.15, stratify=train[target], random_state=17)

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_val = val.drop(columns=[target])
    y_val = val[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    mlflow.set_experiment("churn_best")
    model_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        best_model = lgb.LGBMClassifier(**params)
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        for key, data in {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [X_test, y_test]}.items():
            pred = best_model.predict_proba(data[0])[:,-1]

            auc = roc_auc_score(data[1], pred)
            acc = accuracy_score(data[1], pred>.5)
        
            mlflow.log_metric(f'{key}_auc', auc)
            mlflow.log_metric(f'{key}_acc', acc)

            mlflow.lightgbm.log_model(best_model, "lightgbm_model")
        
    
    best_model.booster_.save_model(os.path.join('models', f'model_{model_name}.txt'))

if __name__ == '__main__':
    train_best_model()



import json
import pandas as pd

import lightgbm as lgb

from datetime import datetime
import os
import argparse
from helper_functions import process_data, split_data, train_and_log_performance

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

with open('config_one.json') as file:
    config = json.load(file)

def train_model(data_name):
    df = pd.read_csv(os.path.join('data', data_name))
    df = process_data(df)

    target = 'Churn_Yes'
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target=target, test_size=0.15)

    params = {}

    for key, value in config.items():
        params[key] = value

    mlflow.set_experiment("churn_one")
    model_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    train_and_log_performance(model_name, params, X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-d", "--data_name", help = "name of the data file", default='data.csv')

    args = parser.parse_args()
    data_name = args.data_name

    train_model(data_name)
        



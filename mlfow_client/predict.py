import lightgbm as lgb
import pandas as pd
import json
import os
from datetime import datetime
from helper_functions import process_data, classification_predictive_power

with open('config_predict.json') as file:
    config = json.load(file)

data_name = config["data"]
model_name = config["model"]

df = pd.read_csv(os.path.join('data', data_name))
df = process_data(df)

model = lgb.Booster(model_file=os.path.join('models', model_name))

target = 'Churn_Yes'

X = df[df.columns.drop(target)]
y = df[target]

pred = pd.DataFrame(model.predict(X), columns=['PRED'])

pred_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
classification_predictive_power(y, pred, name=pred_name)

pred.to_csv(os.path.join('predictions', f'pred_{pred_name}.csv'))
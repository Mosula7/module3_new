import json
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from datetime import datetime
import os
import argparse
from process_data import process_data

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

with open('config_rs.json') as file:
    config = json.load(file)

def train_model():
    pass
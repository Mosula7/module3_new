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

import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, ConfusionMatrixDisplay, classification_report
from scipy.stats import ks_2samp
from operator import itemgetter

import mlflow
import lightgbm as lgb


def process_data(df):
    df_proc = df.drop(columns=['customerID'])
    df_proc['TotalCharges'] = df_proc['TotalCharges'].replace(' ', np.nan).astype('float64')
    df_proc = pd.get_dummies(df_proc, columns=['gender', 'Partner', 'Dependents',
                                            'PhoneService', 'MultipleLines', 
                                            'InternetService', 'OnlineSecurity',
                                            'OnlineBackup', 'DeviceProtection',
                                            'TechSupport', 'StreamingTV', 
                                            'StreamingMovies', 'Contract',   
                                            'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)
    df_proc.columns = [col.replace(' ', '_') for col in df_proc.columns]

    for col in df_proc.columns:
        if df_proc[col].dtype == 'bool':
            df_proc[col] = df_proc[col].astype('int16')

    return df_proc


def split_data(df: pd.DataFrame, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
    """
    returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not val_size:
        val_size = test_size / (1 - test_size)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

    X_train = train[train.columns.drop(target)]
    X_val = val[val.columns.drop(target)]
    X_test = test[test.columns.drop(target)]

    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def classification_predictive_power(y, pred, scoring_func=lambda x: x, name='fig'):
    """
    makes 4 plots:
    first two plots are PDF and CDF of probabilities or scores (if additional scoring/transformation function is provided)
    the third plot is a confusion matrix and the final plot is a classification report
    the function also calculates KS statistic, AUC and accuracy
    """
    title_font_size = 14
    table_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=1,as_cmap=True)
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2,2, figsize=(18,14))

    #PDF
    ks_data = pd.DataFrame({'Target': y, 'prob': pred})
    ks_data['SCORE'] = ks_data['prob'].apply(scoring_func)
    sns.histplot(data=ks_data, x='SCORE', hue='Target',  stat='probability', kde=True, bins=20, common_bins=False, 
                 common_norm=False, palette=['darkorange', 'grey'], ax=ax[0][0], edgecolor='black')
    ax[0][0].set_title('PDF', fontsize=title_font_size)

    #CDF
    sns.kdeplot(data=ks_data,x='SCORE', hue='Target', cumulative=True, common_norm=False, common_grid=True,
                palette=['darkorange', 'grey'], ax=ax[0][1])
    ax[0][1].set_title('CDF', fontsize=title_font_size)

    #confusion matrix
    ConfusionMatrixDisplay.from_predictions(y, pred > 0.5, ax=ax[1][0], cmap=table_cmap)
    ax[1][0].grid(None)
    ax[1][0].set_title('Confusion Matrix', fontsize=title_font_size)

    # classification report
    cr = pd.DataFrame(itemgetter(*[str(i) for i in y.unique()])(classification_report(y, pred>.5, output_dict=True))).drop(columns = 'support')
    sns.heatmap(cr,annot=True,vmax=1,fmt='.5f',ax=ax[1][1],
                cmap=table_cmap)
    ax[1][1].set_title('Classification Report', fontsize=title_font_size)
    
    # KS, AUC, Accuracy
    fig.suptitle(f"""\
                 KS - {ks_2samp(ks_data.query('Target == 0')['SCORE'], ks_data.query('Target == 1')['SCORE'])[0]:.4f}\
                 AUC - {roc_auc_score(y, pred):.4f}\
                 Accuracy - {accuracy_score(y, pred>.5):.4f}""", y=.95, fontsize=16)
    
    fig.savefig(os.path.join('graphs', f'{name}.jpg'))


def train_and_log_performance(model_name, params, X_train, y_train, X_val, y_val, X_test, y_test):
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

            if key == 'test':
                classification_predictive_power(data[1], pred, name=model_name)
        
        best_model.booster_.save_model(os.path.join('models', f'model_{model_name}.txt'))
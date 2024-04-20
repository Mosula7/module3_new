import numpy as np
import pandas as pd

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
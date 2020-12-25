import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

current_folder = os.path.dirname(os.path.abspath(__file__))
project_root = '\\'.join(current_folder.split('\\')[:-1])

def scale(df, scaling = 'robust', colnames = ['Time','Amount']):
    # NOTE: Normalization should be done after train-test split!
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    # Initialize scaler
    if scaling == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    # Scale
    for col in colnames:
        try:
            df_copy['scaled_{}'.format(col.lower())] = scaler.fit_transform(df_copy[col].values.reshape(-1,1))
            df_copy.drop([col], axis=1, inplace=True)
        except:
            print('Cannot scale "{}". The variable might not exist.'.format(col))
    return df_copy

def data_split(df, target = 'Class', ratio = 0.75, random_state = 27):
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Separate features and target
    y = df[target]
    X = df.drop(target, axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-ratio, random_state=27)
        
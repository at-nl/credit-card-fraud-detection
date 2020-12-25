import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import ADASYN, SMOTE

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

def data_split(df, target = 'Class', test_size = 0.25, random_state = 27):
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Separate features and target
    y = df_copy[target]
    X = df_copy.drop(target, axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state
    )
    return (X_train, X_test, y_train, y_test)

def oversample(X_train, y_train, target = 'Class', random_state = 27):
    X = pd.concat([X_train, y_train], axis = 1)
    not_fraud = X[X[target] == 0]
    fraud = X[X[target] == 1]
    # Upsample minority
    fraud_upsampled = resample(
        fraud,
        replace=True, # sample with replacement
        n_samples = len(not_fraud), # match number in majority class
        random_state = random_state # reproducible results
    )

    # Combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])
    
    # Output X and y again
    y_out = upsampled[target]
    X_out = upsampled.drop(target, axis=1)
    
    return (X_out, y_out)

def undersample(X_train, y_train, target = 'Class', random_state = 27):
    X = pd.concat([X_train, y_train], axis = 1)
    not_fraud = X[X[target] == 0]
    fraud = X[X[target] == 1]
    
    # downsample majority
    not_fraud_downsampled = resample(
        not_fraud,
        replace = False, # sample without replacement
        n_samples = len(fraud), # match minority n
        random_state = 27 # reproducible results
    )

    # combine minority and downsampled majority
    downsampled = pd.concat([not_fraud_downsampled, fraud])
    
    # Output X and y again
    y_out = downsampled[target]
    X_out = downsampled.drop(target, axis=1)
    
    return (X_out, y_out)

def synthesize_samples(X_train, y_train, method = 'SMOTE', random_state = 27):
    if method == 'SMOTE':
        syn = SMOTE(random_state = random_state)
    else:
        syn = ADASYN(random_state = random_state)
    X_out, y_out = syn.fit_sample(X_train, y_train)
    return (X_out, y_out)
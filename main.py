import warnings
warnings.filterwarnings('ignore')
from utils import get_data, generate_df_report
from utils.preprocess import scale,data_split,oversample,undersample,synthesize_samples
from models.logistic import Logistic, Logistic_tuned
from models.randomForest import RandomForest, RandomForest_tuned
import pandas as pd
import numpy as np

def main():
    # df = get_data()
    df = pd.read_csv(r'.\data\creditcard.csv')
    # generate_df_report(df, title = 'data_report_2')
    
    # Train-test split
    X_train, X_test, y_train, y_test = data_split(df)
    
    # Scale
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    
    # Oversampling
    over_X_train, over_y_train = oversample(X_train_scaled, y_train)
    
    # # Undersampling
    # under_X_train, under_y_train = undersample(X_train_scaled, y_train)
    
    # # Synthetic sampling (SMOTE)
    # syn_X_train1, syn_y_train1 = synthesize_samples(X_train_scaled, y_train)
    
    # # Synthetic sampling (ADASYN)
    # syn_X_train2, syn_y_train2 = synthesize_samples(X_train_scaled, y_train, method = 'ADASYN')
    
    
    model1 = Logistic()
    model1.fit(over_X_train, over_y_train)
    model1.predict(X_test_scaled)
    model1.score(y_test, method = 'all')
    
    model2 = RandomForest()
    model2.fit(over_X_train, over_y_train)
    model2.predict(X_test_scaled)
    model2.score(y_test, method = 'all')

if __name__ == '__main__':
    main()
from utils import get_data, generate_df_report
import pandas as pd
import numpy as np

## Other functions up here

def main():
    # df = get_data()
    df = pd.read_csv(r'.\data\creditcard.csv')
    # generate_df_report(df, title = 'data_report_2')

if __name__ == '__main__':
    main()
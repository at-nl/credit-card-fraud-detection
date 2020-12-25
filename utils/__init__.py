import kaggle
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from pandas_profiling import ProfileReport

current_folder = os.path.dirname(os.path.abspath(__file__))
project_root = '\\'.join(current_folder.split('\\')[:-1])

def get_data():
    '''
    Get the credit card dataset from Kaggle.
    Link: https://www.kaggle.com/mlg-ulb/creditcardfraud
    
    Args:
        None
    Returns:
        df: dataframe containing credit card data
    '''
    kaggle.api.authenticate() 
    kaggle.api.dataset_download_files(
        'mlg-ulb/creditcardfraud',
        path = project_root + r'\data',
        unzip=True
    )
    file_name = [
        f for f in listdir(
            project_root + r'\data'
        ) if isfile(
            join(
                project_root + r'\data',
                f
            )
        )
    ][0]
    df = pd.read_csv(project_root + '\\data\\' + file_name)
    return df

def generate_df_report(df, exploratory = False, title = 'data_report'):
    '''
    Generates an exploratory analysis report of a Pandas dataframe.
    
    Args:
        - df: an input Pandas dataframe
    Returns:
        - None
    '''
    profile = ProfileReport(
        df,
        title='Pandas Profiling Report',
        explorative = exploratory
    )
    # Generate HTML report page
    profile.to_file(project_root + '\\data\\{}.html'.format(title))
    
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
import os
import logging
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler= logging.StreamHandler()
file_handler = logging.FileHandler('logs/feature_engineered.log',mode='w')
formatter= logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
def load_data(data_url):
    logger.info('data loading started')
    try:
        df= pd.read_csv(data_url)
        logger.debug('data loading completed')
        return df
    except Exception as e:
        logger.error('url is not correct')

def feature_engineer(df:pd.DataFrame):
    logger.debug('Feature engineering just started bro')
    try:
        df['change_in_direction'] = abs(df['winddirection'] - df['winddirection'].shift(1))
        pt = PowerTransformer(method='yeo-johnson')
        df['pressure'] = pt.fit_transform(df[['pressure']])
        df=df.fillna(0)
        logger.debug('feature engineering completed')
        return df
    except Exception as e:
        logger.error(f'ye dikkat h{e}')
        
def save_data(df: pd.DataFrame):
    """
    Saves the DataFrame to 'data/preprocessed_data/p_data.csv' with no index.
    Creates the directory if it doesn't exist.
    """
    logger.debug('Saving process started')
    try:
        os.makedirs('data/featured_data', exist_ok=True)
        df.to_csv('data/featured_data/f_data.csv', index=False)
        logger.debug('Data successfully saved at data/preprocessed_data/p_data.csv')
    except Exception as e:
        logger.error(f'Unknown problem occurred while saving data: {e}')



def main():
    df=load_data(r'C:\Users\DELL\OneDrive\Desktop\MLOps_first_Picture_of_AWS\data\preprocessed_data\p_data.csv')
    df=feature_engineer(df)
    save_data(df)

if __name__ == '__main__':
    main()
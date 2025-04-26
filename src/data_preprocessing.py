import pandas as pd 
import numpy as np
import os
import logging
logger= logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler= logging.StreamHandler()
file_handler= logging.FileHandler('logs/datapreprocessing.log',mode='w')
formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def data_loader(path):
    try:
        df=pd.read_csv(path)
        logger.debug('data easily loaded')
        return df
    except Exception as e:
        logger.error('some problem occured')
        
    
    
def processor(df:pd.DataFrame):
    logger.debug('Preprocessing just strarted')
    try:
        df=df.drop('id',axis=1)
        for i in df.columns:
            df[i]=df[i].fillna(df[i].mean())
        logger.debug('Preprocessing successfully completed')
        return df
    except ValueError as e:
        logger.error('Check the values')
    except Exception as e:
        logger.error('Unknown problem occured')
 
 
        
def save_data(df: pd.DataFrame):
    """
    Saves the DataFrame to 'data/preprocessed_data/p_data.csv' with no index.
    Creates the directory if it doesn't exist.
    """
    logger.debug('Saving process started')
    try:
        os.makedirs('data/preprocessed_data', exist_ok=True)
        df.to_csv('data/preprocessed_data/p_data.csv', index=False)
        logger.debug('Data successfully saved at data/preprocessed_data/p_data.csv')
    except Exception as e:
        logger.error(f'Unknown problem occurred while saving data: {e}')

    
    
def main():
    df=data_loader(r'C:\Users\DELL\OneDrive\Desktop\MLOps_first_Picture_of_AWS\data\raw\raw_data.csv')
    df=processor(df)
    save_data(df)

if __name__ == '__main__':
    main()
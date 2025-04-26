import pandas as pd
import numpy as np
import logging
import os
url=r'https://raw.githubusercontent.com/GitExplorer88/data/refs/heads/main/train.csv'
os.makedirs('logs',exist_ok=True)
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler=logging.StreamHandler()
file_handler=logging.FileHandler('logs/raw.log',mode='a')
formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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

def save_data(df:pd.DataFrame):
    try:
        os.makedirs('data/raw',exist_ok=True)
        logger.debug('folder created')
        df.to_csv('data/raw/raw_data.csv',index=True)
        logger.debug('data saved under the folder')
    except Exception as e:
        logger.error('code has some problem')
    
def main():
    df=load_data(url)
    print(df.head())
    save_data(df)
    
    
if __name__ == '__main__':
    main()
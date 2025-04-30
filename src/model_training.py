import pandas as pd
import numpy as np
import os
import logging
import pickle 
from sklearn.linear_model import LogisticRegression
import yaml


os.makedirs('logs',exist_ok=True)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
stream_handler=logging.StreamHandler()
file_handler=logging.FileHandler('logs/model_train.log',mode='w')
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
        return None

def get_model(x:np.ndarray,y:np.ndarray,params:dict)->LogisticRegression:
    logger.info('Data goes to model')
    try:
        model=LogisticRegression(penalty=params['penalty'],C=params['C'],solver=params['solver'])
        logger.debug('Now model completly ready to be trained')
        model.fit(x,y)
        logger.debug('model trained')
        return model
    except Exception as e:
        logger.error(f'Unknown problem occured{e}')
        
def save_model(model,filepath:str)-> None:
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    with open (filepath,'wb') as f:
        pickle.dump(model,f)
        
        
        
def main():
    df=load_data(r'C:\Users\DELL\MLOps_first_Picture_of_AWS\data\featured_data\f_data.csv')
    x=df.drop('rainfall', axis=1)
    y=df['rainfall']
    with open ('params.yaml') as f:
        p=yaml.safe_load(f)
        
    params = p['model_building']
    file_path='model/model.pkl'
    model=get_model(x,y,params)
    save_model(model,file_path)
    
if __name__ == '__main__':
    main()
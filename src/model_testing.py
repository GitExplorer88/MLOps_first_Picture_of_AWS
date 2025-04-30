import pandas as pd
import numpy as np
import os
import logging
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live
import yaml  

# Logging setup
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/model_evaluation.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Load model
def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            logger.debug('Model successfully loaded')
        return model
    except Exception as e:
        logger.error(f"{e} = this problem occurred")

# Load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data successfully loaded')
        return df
    except Exception as e:
        logger.error(f'{e} = this problem occurred')

# Evaluation
def evaluation(model, df: pd.DataFrame, target: str):
    try:
        x = df.drop(target, axis=1)
        y_test = df[target]
        y_pred = model.predict(x)
        y_pred_proba = model.predict_proba(x)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

# Save metrics
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug('Evaluation metrics saved to JSON')
    except Exception as e:
        logger.error('Error saving metrics: %s', e)

# Example usage
if __name__ == "__main__":
    
    model = load_model(r'C:\Users\DELL\OneDrive\Desktop\MLOps_first_Picture_of_AWS\model\model.pkl')
    df = load_data(r'C:\Users\DELL\OneDrive\Desktop\MLOps_first_Picture_of_AWS\data\featured_data\f_data.csv')
    metrics = evaluation(model, df, target='rainfall')
    y_real= df['rainfall']
    x=df.drop(['rainfall'],axis=1)
    y_test=model.predict(x)
    with open ('params.yaml') as f:
        p=yaml.safe_load(f)
        
    params = p['model_building']
    
    with Live(save_dvc_exp=True) as live:
        live.log_metric('accuracy', accuracy_score(y_real,y_test))
        live.log_metric('precision',precision_score(y_real,y_test))
        live.log_metric('recall', recall_score(y_test, y_test))
        live.log_params(params)
    
    save_metrics(metrics, 'evaluation_metrics.json')
    

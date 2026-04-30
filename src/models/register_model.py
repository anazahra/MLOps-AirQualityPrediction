"""
LK-07: Model Registry dan Versioning
=====================================
Mendaftarkan model terbaik LK-06 ke MLflow Model Registry,
membuat versi kedua, dan melakukan transisi stage.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from glob import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset():
    if os.path.exists('data/raw/dataset_combined_kedua.csv'):
        return pd.read_csv('data/raw/dataset_combined_kedua.csv')
    files = glob('data/raw/dataset_combined*.csv')
    if not files:
        raise FileNotFoundError('Dataset tidak ditemukan!')
    return pd.read_csv(max(files, key=os.path.getctime))

def prepare_features(df):
    from sklearn.preprocessing import LabelEncoder
    drop_cols = ['city', 'timestamp', 'aqi_category', 'latitude', 'longitude']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df['aqi_category'])
    return X, y, le

def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    mlflow.set_experiment('AQI_Prediction_LK07')
    client = MlflowClient()
    MODEL_NAME = 'AQI_RandomForest'

    df = load_dataset()
    X, y, le = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ============================================
    # VERSI 2: Parameter sedikit berbeda dari LK-06
    # (v1 sudah ada dari LK-06, ini kita buat v2)
    # ============================================
    logger.info('=== Training model untuk versi 2 ===')
    params_v2 = {'n_estimators': 300, 'max_depth': 12, 'max_features': 'sqrt'}

    with mlflow.start_run(run_name='RF_v2_LK07_n300_d12'):
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_param('n_estimators', params_v2['n_estimators'])
        mlflow.log_param('max_depth', params_v2['max_depth'])
        mlflow.log_param('max_features', params_v2['max_features'])
        mlflow.log_param('random_state', 42)
        mlflow.log_param('version_label', 'v2_LK07')

        clf = RandomForestClassifier(
            n_estimators=params_v2['n_estimators'],
            max_depth=params_v2['max_depth'],
            max_features=params_v2['max_features'],
            random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        mlflow.log_metric('accuracy', round(acc, 4))
        mlflow.log_metric('f1_macro', round(f1, 4))
        mlflow.log_metric('precision_macro', round(precision, 4))
        mlflow.log_metric('recall_macro', round(recall, 4))

        mlflow.sklearn.log_model(
            clf,
            artifact_path='model',
            registered_model_name=MODEL_NAME  # otomatis jadi v2
        )
        run_id_v2 = mlflow.active_run().info.run_id
        logger.info(f'[v2] Acc: {acc:.4f} | F1: {f1:.4f} | Run ID: {run_id_v2}')

    # Tampilkan semua versi yang terdaftar
    print('\n' + '='*60)
    print('DAFTAR VERSI MODEL DI REGISTRY')
    print('='*60)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        print(f'  Versi {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id[:8]}...')
    print('='*60)
    print('\nLanjutkan: transisi stage')

if __name__ == '__main__':
    main()
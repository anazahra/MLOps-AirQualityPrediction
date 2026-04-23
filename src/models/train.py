"""
AQI Model Training Script dengan MLflow
=========================================
Melatih Random Forest dan XGBoost untuk prediksi kategori AQI.
Setiap run dicatat di MLflow untuk perbandingan eksperimen.

Cara menjalankan:
  python src/models/train.py

Untuk melihat hasil di MLflow UI:
  mlflow ui
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset():
    """Load dataset PRIORITAS: combined_kedua > features > combined"""
    # PRIORITAS 1: Dataset lengkap 601 rows
    if os.path.exists('data/raw/dataset_combined_kedua.csv'):
        latest = 'data/raw/dataset_combined_kedua.csv'
        logger.info(f'✅ PRIORITY: {latest}')
        df = pd.read_csv(latest)
        logger.info(f'Dataset: {df.shape[0]} records, {df.shape[1]} kolom')
        return df
    
    # Fallback lama
    files = glob('data/processed/features_*.csv')
    if not files:
        files = glob('data/raw/dataset_combined*.csv')
    if not files:
        raise FileNotFoundError('Tidak ada dataset!')
    
    latest = max(files, key=os.path.getctime)
    logger.info(f'Memuat dataset: {latest}')
    df = pd.read_csv(latest)
    logger.info(f'Dataset: {df.shape[0]} records, {df.shape[1]} kolom')
    return df


def prepare_features(df):
    """Persiapkan fitur dan label untuk training"""
    # Kolom yang tidak dipakai sebagai fitur
    drop_cols = ['city', 'timestamp', 'aqi_category', 'latitude', 'longitude']
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Hapus kolom non-numerik lainnya
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    # Label encoding untuk target
    le = LabelEncoder()
    y = le.fit_transform(df['aqi_category'])

    logger.info(f'Fitur: {X.shape[1]} kolom')
    logger.info(f'Kelas: {list(le.classes_)}')
    return X, y, le


def train_random_forest(X_train, y_train, X_test, y_test, le, params):
    """Training Random Forest dengan MLflow logging"""
    model_name = f"RF_n{params['n_estimators']}_d{params['max_depth']}_f{params['max_features']}"

    with mlflow.start_run(run_name=model_name):
        # Log parameter
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_param('max_depth', params['max_depth'])
        mlflow.log_param('max_features', params['max_features'])
        mlflow.log_param('random_state', 42)

        # Train
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        # Log metrik
        mlflow.log_metric('accuracy', round(acc, 4))
        mlflow.log_metric('f1_macro', round(f1, 4))
        mlflow.log_metric('precision_macro', round(precision, 4))
        mlflow.log_metric('recall_macro', round(recall, 4))

        # Log model
        mlflow.sklearn.log_model(
            clf,
            artifact_path='model',
            registered_model_name='AQI_RandomForest'
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f'[RF] {model_name} | Acc: {acc:.4f} | F1: {f1:.4f} | Run ID: {run_id}')
        return {'model': clf, 'run_id': run_id, 'accuracy': acc, 'f1': f1, 'name': model_name, 'type': 'RandomForest'}


def train_xgboost(X_train, y_train, X_test, y_test, le, params):
    """Training XGBoost dengan MLflow logging"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning('XGBoost tidak terinstall. Skip XGBoost run.')
        return None

    model_name = f"XGB_lr{params['learning_rate']}_n{params['n_estimators']}_d{params['max_depth']}"

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param('model_type', 'XGBoost')
        mlflow.log_param('learning_rate', params['learning_rate'])
        mlflow.log_param('n_estimators', params['n_estimators'])
        mlflow.log_param('max_depth', params['max_depth'])

        clf = XGBClassifier(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0
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
            registered_model_name='AQI_XGBoost'
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f'[XGB] {model_name} | Acc: {acc:.4f} | F1: {f1:.4f} | Run ID: {run_id}')
        return {'model': clf, 'run_id': run_id, 'accuracy': acc, 'f1': f1, 'name': model_name, 'type': 'XGBoost'}


def main():
    logger.info('=== MULAI TRAINING LK-06 ===')

    # Set experiment name
    mlflow.set_experiment('AQI_Prediction_LK06')

    # Load dan siapkan data
    df = load_dataset()
    X, y, le = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None if len(np.unique(y)) > 1 else None
    )
    logger.info(f'Train: {len(X_train)} | Test: {len(X_test)}')

    results = []

    # ============================================
    # RUN 1: Random Forest - Baseline
    # ============================================
    logger.info('--- Run 1: Random Forest Baseline ---')
    r = train_random_forest(X_train, y_train, X_test, y_test, le, {
        'n_estimators': 100,
        'max_depth': 5,
        'max_features': 'sqrt'
    })
    if r: results.append(r)

    # ============================================
    # RUN 2: Random Forest - Lebih dalam
    # ============================================
    logger.info('--- Run 2: Random Forest Deep ---')
    r = train_random_forest(X_train, y_train, X_test, y_test, le, {
        'n_estimators': 200,
        'max_depth': 10,
        'max_features': 'sqrt'
    })
    if r: results.append(r)

    # ============================================
    # RUN 3: Random Forest - Max features berbeda
    # ============================================
    logger.info('--- Run 3: Random Forest Log2 Features ---')
    r = train_random_forest(X_train, y_train, X_test, y_test, le, {
        'n_estimators': 150,
        'max_depth': 8,
        'max_features': 'log2'
    })
    if r: results.append(r)

    # ============================================
    # RUN 4: XGBoost
    # ============================================
    logger.info('--- Run 4: XGBoost ---')
    r = train_xgboost(X_train, y_train, X_test, y_test, le, {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 6
    })
    if r: results.append(r)

    # ============================================
    # RINGKASAN HASIL
    # ============================================
    if results:
        print('\n' + '='*60)
        print('RINGKASAN EKSPERIMEN LK-06')
        print('='*60)
        print(f'{"Run":<30} {"Accuracy":>10} {"F1-Macro":>10}')
        print('-'*60)
        for r in results:
            print(f"{r['name']:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
        best = max(results, key=lambda x: x['f1'])
        print('='*60)
        print(f"BEST MODEL: {best['name']}")
        print(f"  Accuracy : {best['accuracy']:.4f}")
        print(f"  F1-Macro : {best['f1']:.4f}")
        print(f"  Run ID   : {best['run_id']}")
        print('='*60)
        print('\nJalankan: mlflow ui')
        print('Lalu buka: http://localhost:5000')


if __name__ == "__main__":
    main()
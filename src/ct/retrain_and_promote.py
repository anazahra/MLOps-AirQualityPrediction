"""
Retraining model + evaluasi komparatif + promosi ke Production.
Model baru hanya dipromosikan jika performanya lebih baik dari model lama
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME      = 'AQI_RandomForest'
EXPERIMENT_NAME = 'AQI_CT_Retraining'

def normalize_labels(df):
    """
    Normalisasi semua label ke sistem aturan baru dari clustering.
    Label BMKG lama di-mapping ke label aturan baru berdasarkan threshold PM2.5 hasil clustering
    """
    def pm25_to_new_category(pm25):
        if pm25 < 36.885:
            return 'Baru_Baik'
        elif pm25 < 63.045:
            return 'Baru_Sedang'
        elif pm25 < 85.3:
            return 'Baru_TidakSehat'
        else:
            return 'Baru_Berbahaya'

    # Re-derive semua label dari PM2.5 menggunakan aturan baru
    df['aqi_category'] = df['pm2_5'].apply(pm25_to_new_category)

    print('\nDistribusi label setelah normalisasi:')
    print(df['aqi_category'].value_counts().to_string())
    return df

def load_latest_dataset():
    shifted_path = 'data/raw/dataset_shifted.csv'
    main_path    = 'data/raw/dataset_combined_kedua.csv'

    if os.path.exists(shifted_path):
        logger.info(f'Menggunakan dataset shifted: {shifted_path}')
        df_ref = pd.read_csv(main_path)
        df_new = pd.read_csv(shifted_path)
        df     = pd.concat([df_ref, df_new], ignore_index=True)
        df     = df.drop_duplicates()
        logger.info(f'Dataset gabungan: {len(df)} rows')
    else:
        df = pd.read_csv(main_path)
        logger.info(f'Dataset utama: {len(df)} rows')

    # Normalisasi semua label ke sistem aturan baru
    df = normalize_labels(df)
    return df

def prepare_features(df):
    drop_cols = ['city', 'timestamp', 'aqi_category', 'new_category', 'latitude', 'longitude']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
    le = LabelEncoder()
    label_col = 'new_category' if 'new_category' in df.columns else 'aqi_category'
    print(f'Label yang digunakan: {label_col}')
    y = le.fit_transform(df[label_col])
    logger.info(f'Fitur: {X.shape[1]} kolom, Kelas: {list(le.classes_)}')

    # Tampilkan distribusi kelas
    unique, counts = np.unique(y, return_counts=True)
    print('\nDistribusi kelas:')
    for cls, cnt in zip(le.classes_, counts):
        print(f'  {cls:<20} : {cnt} rows')

    return X, y, le

def get_current_production_metrics(client):
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == 'Production']

        if not prod_versions:
            logger.info('Tidak ada model Production saat ini')
            return None, None

        prod_version = prod_versions[0]
        run = mlflow.get_run(prod_version.run_id)
        acc = run.data.metrics.get('accuracy', 0)
        f1  = run.data.metrics.get('f1_macro', 0)

        print(f'Model Production saat ini (v{prod_version.version}):')
        print(f'  Accuracy : {acc:.4f}')
        print(f'  F1-Macro : {f1:.4f}')

        return acc, f1

    except Exception as e:
        logger.warning(f'Tidak bisa ambil metrik Production: {e}')
        return None, None

def retrain_model(X_train, y_train, X_test, y_test, le):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name='CT_Retraining_Auto') as run:
        mlflow.log_param('trigger', 'continuous_training')
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_param('n_estimators', 200)
        mlflow.log_param('max_depth', 10)
        mlflow.log_param('class_weight', 'balanced')   
        mlflow.log_param('stratify', True)              
        mlflow.log_param('dataset_size', len(X_train) + len(X_test))

        # Solusi 2: class_weight='balanced'
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

        mlflow.log_metric('accuracy', round(acc, 4))
        mlflow.log_metric('f1_macro', round(f1, 4))
        mlflow.sklearn.log_model(clf, 'model', registered_model_name=MODEL_NAME)

        run_id = run.info.run_id
        logger.info(f'Model baru: Acc={acc:.4f}, F1={f1:.4f}, Run ID={run_id}')

        print(f'\nClassification Report (Model Baru):')
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    return clf, acc, f1, run_id

def compare_and_promote(client, new_acc, new_f1, old_acc, old_f1, run_id):
    print('\n=== Evaluasi Komparatif ===')
    print(f'{"":30} {"Accuracy":>10} {"F1-Macro":>10}')
    print('-' * 55)

    if old_acc is not None:
        print(f'{"Model Lama (Production)":<30} {old_acc:>10.4f} {old_f1:>10.4f}')
    else:
        print(f'{"Model Lama":30} {"(tidak ada)":>10}')

    print(f'{"Model Baru (CT Result)":<30} {new_acc:>10.4f} {new_f1:>10.4f}')
    print('-' * 55)

    if old_acc is None or (new_f1 >= old_f1 and new_acc >= old_acc):
        print('-> Model BARU lebih baik atau tidak ada model lama')
        print('-> PROMOSI ke Production!')

        versions = client.search_model_versions(f"name='{MODEL_NAME}'", order_by=['version_number DESC'])
        latest_version = versions[0].version

        old_prod = [v for v in versions if v.current_stage == 'Production' and v.version != latest_version]
        for v in old_prod:
            client.transition_model_version_stage(name=MODEL_NAME, version=v.version, stage='Archived')
            print(f'   Model v{v.version} diarsipkan')

        client.transition_model_version_stage(name=MODEL_NAME, version=latest_version, stage='Production')
        print(f'   Model v{latest_version} dipromosikan ke Production!')
        return True

    else:
        print('-> Model BARU tidak lebih baik dari model lama')
        print('-> Model lama tetap di Production, model baru tidak dipromosikan')
        return False

def main():
    print('=' * 60)
    print('CT Retraining & Promotion')
    print('=' * 60)

    client = MlflowClient()

    # 1. Ambil metrik model Production saat ini
    old_acc, old_f1 = get_current_production_metrics(client)

    # 2. Load dataset terbaru
    df = load_latest_dataset()
    X, y, le = prepare_features(df)

    # Solusi 3: stratify=y supaya test set representatif semua kelas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # ← proporsional semua kelas di train & test
    )
    logger.info(f'Train: {len(X_train)} rows, Test: {len(X_test)} rows')

    # 3. Retrain model
    clf, new_acc, new_f1, run_id = retrain_model(X_train, y_train, X_test, y_test, le)

    # 4. Bandingkan dan putuskan promosi
    promoted = compare_and_promote(client, new_acc, new_f1, old_acc, old_f1, run_id)

    print('\n=== Ringkasan CT ===')
    print(f'  Model baru Accuracy : {new_acc:.4f}')
    print(f'  Model baru F1-Macro : {new_f1:.4f}')
    print(f'  Dipromosikan        : {"Ya" if promoted else "Tidak"}')
    print('=' * 60)

    sys.exit(0)

if __name__ == '__main__':
    main()
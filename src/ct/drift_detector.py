"""
Deteksi data drift menggunakan Kolmogorov-Smirnov test.
Membandingkan distribusi data referensi (training) vs data baru
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fitur yang dipantau untuk drift
MONITORED_FEATURES = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3', 'aqi']

# Threshold p-value: jika p < threshold -> drift terdeteksi
DRIFT_THRESHOLD = 0.05

def load_reference_data():
    """Load data referensi (dataset training yang sudah ada)"""
    ref_path = 'data/raw/dataset_combined_kedua.csv'
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f'Data referensi tidak ditemukan: {ref_path}')
    df = pd.read_csv(ref_path)
    logger.info(f'Data referensi: {len(df)} rows')
    return df

def load_new_data():
    """
    Load data baru yang masuk.
    Dalam simulasi, kita gunakan file shifted data.
    Dalam production, ini bisa dari endpoint API atau folder monitoring
    """
    new_path = 'data/raw/dataset_shifted.csv'
    if not os.path.exists(new_path):
        logger.warning('Data baru tidak ditemukan, gunakan referensi sebagai fallback.')
        return None
    df = pd.read_csv(new_path)
    logger.info(f'Data baru: {len(df)} rows')
    return df

def detect_drift(df_reference, df_new, threshold=DRIFT_THRESHOLD):
    """
    Jalankan KS-test untuk setiap fitur yang dipantau.
    Return: dict hasil dan boolean apakah drift terdeteksi
    """
    results = {}
    drift_detected = False

    print('\n=== Hasil Deteksi Data Drift ===')
    print(f'Threshold p-value: {threshold}')
    print(f'{"Fitur":<12} {"KS-Stat":>10} {"p-value":>12} {"Status":>12}')
    print('-' * 50)

    for feature in MONITORED_FEATURES:
        if feature not in df_reference.columns or feature not in df_new.columns:
            continue

        ref_vals = df_reference[feature].dropna().values
        new_vals = df_new[feature].dropna().values

        ks_stat, p_value = stats.ks_2samp(ref_vals, new_vals)

        is_drift = p_value < threshold
        if is_drift:
            drift_detected = True

        status = 'DRIFT!' if is_drift else 'OK'
        print(f'{feature:<12} {ks_stat:>10.4f} {p_value:>12.6f} {status:>12}')

        results[feature] = {
            'ks_stat': round(ks_stat, 4),
            'p_value': round(p_value, 6),
            'drift': is_drift
        }

    print('=' * 50)
    if drift_detected:
        drifted = [f for f, r in results.items() if r['drift']]
        print(f'Drift Terdeteksi pada: {drifted}')
        print('-> Trigger retraining!')
    else:
        print('Tidak ada drift signifikan')
        print('-> Retraining tidak diperlukan dari sisi data')

    return results, drift_detected

def main():
    df_ref = load_reference_data()
    df_new = load_new_data()

    if df_new is None:
        print('Tidak ada data baru untuk dibandingkan')
        return False, {}

    results, drift_detected = detect_drift(df_ref, df_new)
    return drift_detected, results

if __name__ == '__main__':
    main()
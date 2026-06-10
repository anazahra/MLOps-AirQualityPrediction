"""
Evaluasi semua kondisi trigger untuk Continuous Training
Skenario A: performance-based (accuracy/F1 di bawah threshold)
Skenario B: data drift-based (KS-test)
Skenario C: schedule-based (dihandle oleh cron di GitHub Actions)
"""
import mlflow
import pandas as pd
import sys
import os
import logging
from drift_detector import load_reference_data, load_new_data, detect_drift

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# THRESHOLD TRIGGER
# ============================================
THRESHOLD_ACCURACY = 0.90   
THRESHOLD_F1       = 0.85
EXPERIMENT_NAME    = 'AQI_Prediction'

def check_performance_trigger():
    """
    Skenario A: Cek apakah performa model terbaru di bawah threshold.
    Ambil metrik dari MLflow run terbaru
    """
    print('\n=== Skenario A: Performance Check ===')

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f'Experiment {EXPERIMENT_NAME} tidak ditemukan')
            return False

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['start_time DESC'],
            max_results=1
        )

        if runs.empty:
            print('Tidak ada run ditemukan')
            return False

        latest = runs.iloc[0]
        acc = latest.get('metrics.accuracy', 1.0)
        f1  = latest.get('metrics.f1_macro', 1.0)

        print(f'Run terbaru: ')
        print(f'  Accuracy : {acc:.4f}  (threshold: {THRESHOLD_ACCURACY})')
        print(f'  F1-Macro : {f1:.4f}  (threshold: {THRESHOLD_F1})')

        trigger = acc < THRESHOLD_ACCURACY or f1 < THRESHOLD_F1
        if trigger:
            print('-> Trigger AKTIF: performa di bawah threshold!')
        else:
            print('-> Performa masih di atas threshold, tidak perlu retrain')

        return trigger

    except Exception as e:
        logger.warning(f'Tidak bisa cek performance: {e}')
        return False

def check_drift_trigger():
    """Skenario B: Deteksi data drift"""
    print('\n=== Skenario B: Data Drift Check ===')

    try:
        df_ref = load_reference_data()
        df_new = load_new_data()

        if df_new is None:
            print('Tidak ada data baru, skip drift check')
            return False

        _, drift_detected = detect_drift(df_ref, df_new)
        return drift_detected

    except Exception as e:
        logger.warning(f'Tidak bisa cek drift: {e}')
        return False

def main():
    """
    Evaluasi semua trigger. Jika salah satu aktif, exit code 0 (trigger retraining).
    Jika tidak ada yang aktif, exit code 1 (skip retraining)
    """
    print('=' * 60)
    print('CT Trigger Evaluation')
    print('=' * 60)

    trigger_A = check_performance_trigger()
    trigger_B = check_drift_trigger()

    # Skenario C (schedule) dihandle langsung oleh cron di GitHub Actions
    # Tidak perlu dicek di sini

    should_retrain = trigger_A or trigger_B

    print('\n=== Final Decision ===')
    print(f'  Skenario A (Performance) : {"Aktif" if trigger_A else "Tidak Aktif"}')
    print(f'  Skenario B (Data Drift)  : {"Aktif" if trigger_B else "Tidak Aktif"}')
    print(f'  Keputusan                : {"Jalankan Retraining" if should_retrain else "Skip"}')
    print('=' * 60)

    # Exit code 0 = retraining diperlukan
    # Exit code 1 = skip
    sys.exit(0 if should_retrain else 1)

if __name__ == '__main__':
    main()
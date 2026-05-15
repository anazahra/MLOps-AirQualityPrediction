"""
LK-08: Evaluasi Otomatis dan Promosi Model ke MLflow Registry

Skrip ini:
1. Mengambil hasil run training terbaru dari MLflow
2. Membandingkan metrik dengan threshold yang ditentukan
3. Jika lolos → daftarkan model ke registry dengan stage Staging
4. Jika tidak lolos → exit dengan kode error (pipeline gagal)
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import sys
import os

EXPERIMENT_NAME   = 'AQI_Prediction'   # Nama experiment MLflow dari LK-06
MODEL_NAME        = 'AQI_RandomForest'       # Nama model di registry
THRESHOLD_ACCURACY = 0.90                    # Threshold dari LK-01
THRESHOLD_F1       = 0.85                    # Threshold F1-macro
# ═══════════════════════════════════════════════════

def evaluate_and_promote():
    print('=' * 60)
    print('EVALUASI DAN PROMOSI MODEL OTOMATIS')
    print('=' * 60)

    client = MlflowClient()

    # 1. Ambil experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f'ERROR: Experiment "{EXPERIMENT_NAME}" tidak ditemukan!')
        sys.exit(1)

    # 2. Ambil run training terbaru
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['start_time DESC'],
        max_results=1
    )

    if runs.empty:
        print('ERROR: Tidak ada run ditemukan!')
        sys.exit(1)

    latest_run = runs.iloc[0]
    acc    = latest_run.get('metrics.accuracy', 0)
    f1     = latest_run.get('metrics.f1_macro', 0)
    run_id = latest_run['run_id']

    print(f'Run ID terbaru : {run_id}')
    print(f'Accuracy       : {acc:.4f}  (Threshold: {THRESHOLD_ACCURACY})')
    print(f'F1-macro       : {f1:.4f}  (Threshold: {THRESHOLD_F1})')
    print('-' * 60)

    # 3. Evaluasi threshold
    if acc >= THRESHOLD_ACCURACY and f1 >= THRESHOLD_F1:
        print('[PASS] Model memenuhi threshold!')

        # 4. Daftarkan ke MLflow Registry → Staging
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f'runs:/{run_id}/model',
            run_id=run_id
        )
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage='Staging'
        )
        print(f'[OK] Model v{mv.version} berhasil dipromosikan ke Staging!')
        print('=' * 60)
    else:
        print('[FAIL] Performa model TIDAK memenuhi threshold!')
        print('       Model TIDAK dipromosikan ke Staging.')
        print('=' * 60)
        sys.exit(1)  # ← Exit code 1 → job registry GAGAL di GitHub Actions

if __name__ == '__main__':
    evaluate_and_promote()
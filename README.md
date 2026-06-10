# 🌫️ MLOps Air Quality Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)](https://mlflow.org)
[![CI/CD Status](https://github.com/anazahra/MLOps-AirQualityPrediction/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/anazahra/MLOps-AirQualityPrediction/actions/workflows/mlops_pipeline.yml)

Sistem prediksi kategori kualitas udara (AQI) berbasis MLOps untuk 10 kota besar Indonesia. Data diambil otomatis dari OpenWeatherMap API setiap jam, diproses melalui pipeline ML end-to-end, dan disajikan melalui REST API dengan monitoring terintegrasi.

## Gambaran Sistem

Sistem ini membangun pipeline MLOps semi-otomatis yang mampu:

- Mengambil data polutan udara (PM2.5, PM10, NO2, O3, CO, SO2) dari API publik setiap jam
- Memprediksi kategori AQI — **Baik / Sedang / Tidak Sehat / Berbahaya** — secara real-time
- Melakukan retraining otomatis ketika performa model turun atau data drift terdeteksi
- Menyajikan prediksi melalui REST API yang dapat di-scale horizontal

```
OpenWeatherMap API + BMKG
          │
          ▼
  [Data Ingestion]  ──→  data/raw/
          │
          ▼
  [Preprocessing]   ──→  data/processed/
          │
          ▼
  [Model Training]  ──→  MLflow Experiment Tracking
          │
          ▼
  [Model Registry]  ──→  Staging → Production
          │
          ▼
  [API Serving]     ──→  FastAPI + MLflow Model Server (x3 replika)
          │
          ▼
  [Monitoring]      ──→  Prometheus + Grafana
```

---

## Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.11 |
| ML Framework | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Serving | FastAPI |
| Containerisasi | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Data Source | OpenWeatherMap API, BMKG Open Data |

---

## Struktur Direktori

```
MLOps-AirQualityPrediction/
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       ├── mlops_pipeline.yml       # CI/CD utama (testing + training)
│       └── ct-pipeline.yml          # Continuous Training otomatis
├── data/
│   ├── raw/                         # Data mentah per ingestion run
│   ├── processed/                   # Hasil preprocessing & feature engineering
│   ├── batches/
│   └── external/
├── models/
│   ├── trained/
│   ├── evaluation/
│   └── model_production.yaml        # Metadata model aktif (tracked DVC)
├── notebooks/
│   └── 01_initial_eda.ipynb
├── src/
│   ├── ingest_data.py               # Ambil data dari API
│   ├── preprocess.py                # Cleaning + feature engineering
│   ├── models/
│   │   ├── train.py                 # Training + MLflow logging
│   │   ├── register_model.py        # Registrasi versi model baru
│   │   └── verify_inference.py      # Verifikasi model Production
│   ├── clustering/
│   │   └── cluster_analysis.py      # K-Means + aturan klasifikasi baru
│   ├── ct/
│   │   ├── drift_detector.py        # Deteksi data drift (KS-test)
│   │   ├── ct_trigger.py            # Evaluasi trigger retraining
│   │   ├── retrain_and_promote.py   # Retraining + promosi komparatif
│   │   └── simulate_drift.py        # Simulasi data shift untuk testing
│   └── evaluate_and_promote.py      # Evaluasi threshold + promosi ke Staging
├── tests/
│   └── test_pipeline.py
├── config/
│   └── config.yaml
├── reports/
│   └── figures/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Alur Pipeline End-to-End

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INGESTION         src/ingest_data.py                        │
│     Ambil data API → data/raw/data_YYYYMMDD_HHMMSS.csv          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  2. PREPROCESSING     src/preprocess.py                         │
│     Cleaning + feature engineering → data/processed/            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  3. CLUSTERING        src/clustering/cluster_analysis.py        │
│     K-Means (k=4) → temukan pola alami data → aturan baru       │
│     Bandingkan RF label BMKG vs RF label aturan baru            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  4. TRAINING          src/models/train.py                       │
│     Random Forest + XGBoost, semua run dicatat di MLflow        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  5. EVALUASI & REGISTRY   src/evaluate_and_promote.py           │
│     Cek threshold → jika lolos, promosi ke Staging              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  6. SERVING           Docker Compose                            │
│     FastAPI (port 8000) + MLflow Model Server (3 replika)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  7. MONITORING        Prometheus + Grafana                      │
│     Metrik API, latency, error rate -> dashboard di port 8080   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  8. CONTINUOUS TRAINING   src/ct/                               │
│     Trigger otomatis jika performa turun atau drift terdeteksi  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Ingestion

**Script:** `src/ingest_data.py`

Mengambil data polutan udara dari OpenWeatherMap Air Pollution API untuk 10 kota besar Indonesia secara non-destruktif (setiap run menyimpan file baru dengan timestamp).

```bash
# Satu kali run
python src/ingest_data.py

# Kumpulkan lebih banyak data
for i in {1..10}; do python src/ingest_data.py; sleep 2; done

# Gabungkan semua CSV menjadi satu dataset
python3 -c "
import pandas as pd, glob
files = glob.glob('data/raw/data_*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
df = df.drop_duplicates(subset=['city','timestamp'])
df.to_csv('data/raw/dataset_combined.csv', index=False)
print(f'Total: {len(df)} records')
"
```

**Output:** `data/raw/data_YYYYMMDD_HHMMSS.csv` → digabung ke `data/raw/dataset_combined.csv`

**Sumber data:**
- OpenWeatherMap Air Pollution API → PM2.5, PM10, NO2, O3, CO, SO2
- BMKG Open Data → suhu, kelembaban, kecepatan angin
- Coverage: Jakarta, Surabaya, Bandung, Medan, Semarang, Makassar, Palembang, Tangerang, Depok, Bekasi

---

## Preprocessing & Feature Engineering

**Script:** `src/preprocess.py`

Membersihkan data mentah dan menghasilkan fitur siap pakai untuk training.

```bash
python src/preprocess.py
```

**Output:** `data/processed/features_YYYYMMDD.csv`

**Fitur yang dihasilkan:**

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `city` | string | Nama kota |
| `pm2_5` | float | PM2.5 (µg/m³) |
| `pm10` | float | PM10 (µg/m³) |
| `no2`, `o3`, `co`, `so2` | float | Polutan lainnya |
| `aqi` | int | AQI index (1–5) |
| `hour_sin`, `hour_cos` | float | Cyclical encoding jam |
| `pm25_lag_1/2/3` | float | Lag features PM2.5 |
| `pm25_rolling_6h` | float | Rolling mean PM2.5 6 jam |
| `aqi_category` | string | **Target label** (dari aturan BMKG, di-rederive oleh clustering) |

---

## Analisis Clustering & Aturan Klasifikasi Baru

**Script:** `src/clustering/cluster_analysis.py`

Tujuan analisis ini adalah menjawab pertanyaan: **apakah threshold klasifikasi statis BMKG sudah merepresentasikan kondisi riil data kualitas udara 10 kota Indonesia?**

Proses yang dijalankan:
1. Jalankan K-Means dengan berbagai nilai k, tentukan k optimal via Elbow Method → **k=4**
2. Bentuk 4 cluster berdasarkan fitur: `aqi`, `pm2_5`, `pm10`, `no2`, `o3`, `co`, `so2`, `nh3`
3. Analisis profil tiap cluster (rata-rata tiap polutan per cluster)
4. Derivasi aturan baru berbasis threshold PM2.5 hasil clustering
5. Latih dua model RF — satu dengan label BMKG, satu dengan label aturan baru — lalu bandingkan

```bash
python src/clustering/cluster_analysis.py
```

**Aturan klasifikasi baru (berbasis data, bukan BMKG):**

| Label Baru | Threshold PM2.5 | Padanan BMKG |
|---|---|---|
| `Baru_Baik` | < 36.885 µg/m³ | Baik |
| `Baru_Sedang` | 36.885 – 63.045 µg/m³ | Sedang |
| `Baru_TidakSehat` | 63.045 – 85.3 µg/m³ | Tidak Sehat |
| `Baru_Berbahaya` | ≥ 85.3 µg/m³ | Berbahaya |

Threshold ini diturunkan dari centroid tiap cluster, sehingga batas antar kategori mencerminkan pola nyata data — bukan nilai tetap yang ditetapkan secara regulasi.

**Perbandingan hasil klasifikasi (MLflow experiment: `AQI_Clustering_NewRules`):**

| Model | Label yang Digunakan | Accuracy | F1-Macro |
|---|---|---|---|
| RF_BMKG_Label | Aturan statis BMKG | 0.9917 | 0.9147 |
| RF_NewRules_Label | Aturan baru (clustering) | 1.0000 | 1.0000 |

Model yang dilatih dengan **label aturan baru menghasilkan performa lebih tinggi**, karena batas antar kelas yang digunakan saat training konsisten dengan pola yang dipelajari model dari data. Threshold BMKG yang bersifat statis kadang menempatkan data di kelas yang tidak selaras dengan pola distribusi aktual, sehingga menimbulkan ambiguitas di batas antar kelas.

**Output:**

| File | Lokasi | Deskripsi |
|---|---|---|
| `clustering_results.csv` | `data/processed/` | Dataset dengan kolom `cluster_id` dan `new_category` |
| `RF_NewRules_Label` | MLflow | Model RF dilatih dengan label aturan baru |
| `RF_BMKG_Label` | MLflow | Model RF dilatih dengan label BMKG (pembanding) |

Aturan baru ini kemudian dipakai secara konsisten di seluruh pipeline CT (`retrain_and_promote.py`) via fungsi `normalize_labels()` yang me-rederive semua label dari nilai PM2.5 menggunakan threshold hasil clustering.

---

## Pelatihan Model & Experiment Tracking

**Script:** `src/models/train.py`

Melatih beberapa konfigurasi model dan mencatat semua parameter, metrik, serta artifact ke MLflow secara otomatis.

```bash
python src/models/train.py
```

**Model yang dilatih:**

| Run | Model | Parameter |
|---|---|---|
| 1 | Random Forest | n=100, depth=5 |
| 2 | Random Forest | n=200, depth=10 |
| 3 | Random Forest | n=150, depth=8, max_features=log2 |
| 4 | XGBoost | lr=0.1, n=100, depth=6 |

**Model terbaik:** Random Forest dengan `n_estimators=200`, `max_depth=10`
- Accuracy: 0.9917 | F1-Macro: 0.9147 | Precision: 0.996 | Recall: 0.875

**MLflow UI:**
```bash
mlflow ui
# → http://localhost:5000
```

**Analisis Clustering** (`src/clustering/cluster_analysis.py`): K-Means (k=4 via Elbow Method) dijalankan untuk menemukan pola alami data dan membandingkan performa model yang dilatih dengan label BMKG vs label berbasis clustering.

---

## Model Registry & Versioning

**Scripts:** `src/models/register_model.py`, `src/evaluate_and_promote.py`

Setelah training, `evaluate_and_promote.py` secara otomatis mengecek apakah model terbaru memenuhi threshold. Jika ya, model didaftarkan ke MLflow Model Registry dan dipromosikan ke Staging.

```
Training selesai
      │
      ▼
Accuracy ≥ 0.90 & F1 ≥ 0.85?
      │ Ya                  │ Tidak
      ▼                     ▼
Promosi ke Staging     Pipeline gagal (exit code 1)
```

**Transisi stage model** dilakukan melalui MLflow UI:
```
None → Staging → Production
```

**Verifikasi inferensi model Production:**
```bash
python src/models/verify_inference.py
```

**Data versioning dengan DVC:**
```bash
dvc add data/raw/dataset_combined.csv
git add data/raw/dataset_combined.csv.dvc
git commit -m "Update dataset"
dvc push
```

**Model aktif saat ini:**

| Komponen | Detail |
|---|---|
| Nama Model | AQI_RandomForest |
| Stage | Production |
| Accuracy | 0.9917 |
| F1-Macro | 0.9147 |

---

## Continuous Training (CT)

**Scripts:** `src/ct/`

Sistem CT memantau tiga skenario trigger retraining secara otomatis:

| Skenario | Kondisi | Threshold |
|---|---|---|
| A — Performance | Accuracy atau F1 turun | Accuracy < 0.90, F1 < 0.85 |
| B — Data Drift | KS-test signifikan | p-value < 0.05 |
| C — Schedule | Jadwal rutin | Setiap Minggu 08.00 WIB |

**Alur CT:**

```
ct_trigger.py
    ├── check_performance_trigger()   ← Skenario A
    └── check_drift_trigger()         ← Skenario B (via drift_detector.py)
          │ (jika salah satu aktif)
          ▼
retrain_and_promote.py
    ├── load_latest_dataset()         ← gabung data lama + data shifted jika ada
    ├── normalize_labels()            ← rederive label pakai threshold clustering
    ├── retrain_model()               ← RF(n=200, depth=10, class_weight='balanced')
    └── compare_and_promote()         ← bandingkan vs model Production saat ini
          │ Model baru lebih baik?
          ├── Ya  → promosi ke Production, model lama diarsipkan
          └── Tidak → model lama tetap aktif
```

**Menjalankan CT secara manual:**
```bash
# Simulasi data drift
python src/ct/simulate_drift.py

# Cek apakah retraining diperlukan
python src/ct/ct_trigger.py

# Jalankan retraining + evaluasi komparatif + promosi
python src/ct/retrain_and_promote.py
```

---

## CI/CD Automation

**Workflow:** `.github/workflows/mlops_pipeline.yml`

Pipeline berjalan otomatis setiap ada push ke `main` atau setiap hari jam 08.00 WIB.

```
Push / PR ke main  ─atau─  Jadwal harian 08.00 WIB
          │
          ▼
┌─────────────────────────┐
│  JOB 1: CI – Validation │  pytest tests/ -v
└────────────┬────────────┘
             │ (jika semua test PASSED)
             ▼
┌─────────────────────────┐
│  JOB 2: CD – Pipeline   │  ingest → preprocess → upload artifact
│  (hanya di branch main) │
└─────────────────────────┘
```

**Workflow CT** (`.github/workflows/ct-pipeline.yml`) memiliki alur tambahan:

```
┌──────────────┐   ┌──────────────┐    ┌──────────────────────┐
│   testing    │──▶│  training   │──▶ │       registry       │
│   pytest     │   │   train.py   │    │  evaluate_and_       │
│              │   │  + dvc pull  │    │  promote.py          │
└──────────────┘   └──────────────┘    └──────────────────────┘
```

**Trigger yang aktif:**

| Trigger | Kapan Berjalan |
|---|---|
| `push` ke `main` / `feat/**` | Setiap commit baru |
| `pull_request` ke `main` | Saat PR dibuka atau diperbarui |
| `schedule` cron `0 1 * * *` | Setiap hari jam 08.00 WIB |
| `workflow_dispatch` | Manual dari tab Actions |

**GitHub Secrets yang diperlukan:**

| Secret | Keterangan |
|---|---|
| `OPENWEATHER_API_KEY` | API key OpenWeatherMap |
| `DAGSHUB_USERNAME` | Username DagsHub untuk DVC pull |
| `DAGSHUB_TOKEN` | Token autentikasi DagsHub |

---

## Orkestrasi & Serving

**File:** `docker-compose.yml`

Seluruh sistem dijalankan dalam satu network Docker (`ml-network`) dengan tiga service utama.

```
┌────────────────────────────────────────────────────────┐
│                Docker Compose — ml-network             │
│                                                        │
│  [api-service:8000]   [aqi-model-server x3]            │
│         │              port 1234 / 1235 / 1236         │
│         └──────────────────┐                           │
│                    [mlflow-server:5000]                │
│                    [Volume: mlflow-data]               │
└────────────────────────────────────────────────────────┘
```

**Menjalankan seluruh sistem:**
```bash
cp .env.example .env         # isi OPENWEATHER_API_KEY
docker compose up -d --scale aqi-model-server=3

# Akses layanan
# MLflow UI     → http://localhost:5000
# API Inferensi → http://localhost:8000
# Model Server  → http://localhost:1234 / 1235 / 1236
```

**Scale replika secara dinamis:**
```bash
docker compose up -d --scale aqi-model-server=5   # scale up
docker compose up -d --scale aqi-model-server=2   # scale down
```

**Contoh request prediksi:**
```bash
curl -X POST http://localhost:1234/invocations \
     -H 'Content-Type: application/json' \
     -d '{"dataframe_split": {"columns": ["pm2_5", "pm10", "no2", "o3", "co", "so2"], "data": [[12.5, 20.0, 5.0, 80.0, 0.3, 2.0]]}}'
```

**Load model langsung di kode:**
```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model('models:/AQI_RandomForest/Production')
prediction = model.predict(data)
```

---

## Observability & Monitoring

```
[api-service:8000/metrics] ──▶ [prometheus:9090] ──▶ [grafana:8080]
[aqi-model-server x3]
[mlflow-server:5000]
```

```bash
docker compose build api-service
docker compose up -d --scale aqi-model-server=3

# Grafana Dashboard → http://localhost:8080  (admin / admin123)
# Prometheus UI     → http://localhost:9090
# API Metrics       → http://localhost:8000/metrics
```

---

## Cara Menjalankan

### GitHub Codespaces (Direkomendasikan)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anazahra/MLOps-AirQualityPrediction)

1. Klik badge di atas → setup otomatis selesai ~3–5 menit
2. Tambahkan API key sebagai Codespaces Secret: `OPENWEATHER_API_KEY`
3. Verifikasi setup:
   ```bash
   python -c "import sklearn, mlflow, fastapi, streamlit; print('Setup OK!')"
   ```

### Setup Lokal

```bash
git clone https://github.com/anazahra/MLOps-AirQualityPrediction.git
cd MLOps-AirQualityPrediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Menjalankan Pipeline Lengkap

```bash
# 1. Ingest data
python src/ingest_data.py

# 2. Preprocessing
python src/preprocess.py

# 3. Training + tracking
python src/models/train.py

# 4. Evaluasi + promosi
python src/evaluate_and_promote.py

# 5. Jalankan sistem lengkap
docker compose up -d --scale aqi-model-server=3
```

---

## Target Metrik

| Metrik | Target | Keterangan |
|---|---|---|
| Accuracy | ≥ 80% | Performa keseluruhan 4 kelas AQI |
| F1-Score Macro | ≥ 0.78 | Keseimbangan antar kelas |
| Recall (Tidak Sehat) | ≥ 85% | Minimasi false negative kondisi berbahaya |
| Inference Latency | ≤ 100ms | Respons real-time |

---

## Kontributor

**Ana Zahratul Firdausi** — 235150201111049

- GitHub: [anazahra/MLOps-AirQualityPrediction](https://github.com/anazahra/MLOps-AirQualityPrediction)
- CI/CD Actions: [lihat status pipeline](https://github.com/anazahra/MLOps-AirQualityPrediction/actions)
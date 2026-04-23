# 🌫️ MLOps-AirQualityPrediction
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)](https://mlflow.org)
[![Open in Codespaces](https://img.shields.io/badge/GitHub-Codespaces-green?logo=github)](https://codespaces.new/anazahra/MLOps-AirQualityPrediction)
 
> Pipeline MLOps semi-otomatis untuk prediksi kategori kualitas udara (AQI)
> secara real-time menggunakan Random Forest dengan data dari OpenWeatherMap
> API dan BMKG, dilengkapi tracking eksperimen via MLflow.
 
## 🎯 Tujuan Proyek
Kualitas udara yang buruk menyebabkan 7 juta kematian prematur per tahun (WHO, 2024).
Proyek ini membangun sistem prediksi AQI berbasis MLOps yang dapat:
- Memprediksi kategori AQI (Baik/Sedang/Tidak Sehat/Berbahaya) secara real-time
- Mengambil data otomatis setiap jam dari API publik gratis
- Melakukan retraining otomatis saat performa turun atau drift terdeteksi
 
**Target Metrik:**
| Metrik | Target | Alasan |
|--------|--------|--------|
| Accuracy | >= 80% | Performa keseluruhan 4 kelas AQI |
| F1-Score Macro | >= 0.78 | Keseimbangan antar kelas termasuk 'Berbahaya' |
| Recall (Tidak Sehat) | >= 85% | Minimasi miss pada kondisi udara berbahaya |
| Inference Latency | <= 100ms | Respons cepat untuk aplikasi real-time |
 
## 📊 Sumber Data
- **OpenWeatherMap Air Pollution API**: PM2.5, PM10, NO2, O3, CO, SO2
  - Daftar gratis: https://openweathermap.org/api (1.000 call/hari)
- **BMKG Open Data**: Suhu, kelembaban, kecepatan angin
  - Akses langsung tanpa registrasi: https://data.bmkg.go.id
- **Coverage**: 10 kota besar Indonesia — update setiap jam (24/7)
 
## 📁 Struktur Direktori
```
MLOps-AirQualityPrediction/
├── .devcontainer/devcontainer.json  
├── data/
│   ├── raw/         
│   ├── processed/    
│   ├── batches/      
│   └── external/     
├── models/
│   ├── trained/      
│   └── evaluation/   
├── notebooks/
│   └── 01_initial_eda.ipynb  
├── src/
│   ├── data/         
│   ├── features/     
│   ├── models/      
│   └── visualization/ 
├── config/config.yaml  
├── reports/figures/    
├── requirements.txt
└── README.md
```
 
## ⚡ Cara Menjalankan dengan GitHub Codespaces
1. Klik badge berikut untuk membuka lingkungan kerja secara instan:
   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/anazahra/MLOps-AirQualityPrediction)
2. Tunggu setup otomatis selesai (~3-5 menit)
3. Semua dependencies otomatis terinstall dari `requirements.txt`
4. Tambahkan API key sebagai Codespaces Secret: `OPENWEATHER_API_KEY`
5. Verifikasi setup:
   ```bash
   python -c "import sklearn, mlflow, fastapi, streamlit; print('Setup OK!')"
   ```
6. Buka `notebooks/01_initial_eda.ipynb` dan jalankan sel pertama
 
## 🛠️ Setup Lokal (Alternatif)
```bash
git clone https://github.com/anazahra/MLOps-AirQualityPrediction.git
cd MLOps-AirQualityPrediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
 
## 🌳 Branching Strategy (GitHub Flow)
| Branch | Tujuan |
|--------|--------|
| `main` | Kode production-ready, hanya dari Pull Request |
| `feat/initial-eda` | EDA awal data AQI *(sudah di-merge)* |
| `feat/data-pipeline` | Pipeline ingestion & preprocessing (mendatang) |
| `feat/model-training` | Training Random Forest + MLflow (mendatang) |

## 🚀 Data Pipeline — LK-04
 
### Prerequisites
Pastikan sudah install semua dependencies:
```bash
pip install -r requirements.txt
```
 
### 1. Menjalankan Data Ingestion
Skrip ini mengambil data AQI real-time dari OpenWeatherMap API
untuk 10 kota besar Indonesia dan menyimpannya secara non-destruktif.
 
```bash
python src/ingest_data.py
```
 
**Output:** `data/raw/data_YYYYMMDD_HHMMSS.csv`
 
Jalankan beberapa kali untuk mengumpulkan data yang cukup:
```bash
for i in {1..10}; do python src/ingest_data.py; sleep 2; done
```
 
### 2. Menggabungkan Data
Setelah ingestion beberapa kali, gabungkan semua CSV:
```bash
python3 -c "
import pandas as pd, glob
files = glob.glob('data/raw/data_*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
df = df.drop_duplicates(subset=['city','timestamp'])
df.to_csv('data/raw/dataset_combined.csv', index=False)
print(f'Total: {len(df)} records')
"
```
 
### 3. Menjalankan Preprocessing
Skrip ini membersihkan data mentah dan menghasilkan fitur siap pakai.
 
```bash
python src/preprocess.py
```
 
**Output:** `data/processed/features_YYYYMMDD.csv`
 
### Format Output Data
 
| File | Lokasi | Deskripsi |
|------|--------|-----------|
| data_YYYYMMDD_HHMMSS.csv | data/raw/ | Data mentah per ingestion run |
| dataset_combined.csv | data/raw/ | Gabungan semua data mentah |
| features_YYYYMMDD.csv | data/processed/ | Data setelah feature engineering |
 
### Kolom Dataset (features_YYYYMMDD.csv)
 
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| city | string | Nama kota |
| pm2_5 | float | PM2.5 (µg/m³) |
| pm10 | float | PM10 (µg/m³) |
| aqi | int | AQI index (1-5) |
| hour_sin, hour_cos | float | Cyclical encoding jam |
| pm25_lag_1/2/3 | float | Lag features PM2.5 |
| pm25_rolling_6h | float | Rolling mean PM2.5 6 jam |
| aqi_category | string | **Target label** (Baik/Sedang/Tidak Sehat/Berbahaya) |

## 📦 Data Versioning — LK-05

### Prerequisites
Pastikan DVC sudah terinstall:
```bash
pip install dvc
```

### 1. Inisialisasi DVC
```bash
dvc init
mkdir -p /workspaces/dvc-storage
dvc remote add -d myremote /workspaces/dvc-storage
git add .dvc/config
git commit -m "Configure DVC local remote storage"
```

### 2. Tracking Dataset (Version 1)
Mendaftarkan dataset ke DVC agar tidak disimpan langsung di Git:
```bash
dvc add data/raw/dataset_combined.csv
git add data/raw/dataset_combined.csv.dvc
git commit -m "Add dataset to DVC tracking [Version 1]"
dvc push
```

### 3. Update Dataset (Version 2)
Simulasi continual learning — ambil data baru dan buat versi baru:
```bash
for i in {1..3}; do python src/ingest_data.py; sleep 2; done
python3 -c "
import pandas as pd, glob
files = glob.glob('data/raw/data_*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
df = df.drop_duplicates(subset=['city','timestamp'])
df.to_csv('data/raw/dataset_combined.csv', index=False)
print(f'Total: {len(df)} records')
"
dvc add data/raw/dataset_combined.csv
git add data/raw/dataset_combined.csv.dvc
git commit -m "Update dataset: add new ingestion data [Version 2]"
dvc push
```

### 4. Audit Versi
Melihat status dan perbedaan antar versi dataset:
```bash
dvc status
git log --oneline
dvc diff HEAD~1 HEAD
```

### 5. Memulihkan Dataset Versi Lama
```bash
git checkout <commit-hash>
dvc checkout
```

### Format File DVC

| File | Lokasi | Deskripsi |
|------|--------|-----------|
| dataset_combined.csv.dvc | data/raw/ | Pointer metadata Version 1 & 2 |
| .dvc/config | .dvc/ | Konfigurasi remote storage DVC |
| /workspaces/dvc-storage | lokal | Penyimpanan data aktual DVC |

---

## 🧪 Experiment Tracking & Model Training — LK-06

### 📌 Deskripsi

Pada tahap ini, sistem MLOps dikembangkan dengan menambahkan **experiment tracking menggunakan MLflow** untuk memonitor performa model secara sistematis.

Model yang digunakan:

* Random Forest (beberapa konfigurasi)
* XGBoost (sebagai pembanding)

Setiap eksperimen akan mencatat:

* Parameter model
* Metrik evaluasi (Accuracy, F1, Precision, Recall)
* Artifact model
* Run ID

---

### ⚙️ Setup MLflow

Install dependencies:

```bash
pip install mlflow xgboost
```

Jalankan MLflow UI:

```bash
mlflow ui
```

Akses dashboard:

```
http://localhost:5000
```

---

### 📂 Struktur Tambahan

```
models/
├── trained/
├── evaluation/

mlruns/        ← otomatis dari MLflow
src/models/
└── train.py   ← script training LK-06
```

---

### ▶️ Menjalankan Training

```bash
python src/models/train.py
```

Script ini akan:

1. Load dataset terbaru
2. Preprocessing fitur
3. Split data train-test
4. Menjalankan beberapa eksperimen model
5. Logging hasil ke MLflow

---

### 🔁 Eksperimen yang Dijalankan

| Run | Model         | Parameter                         |
| --- | ------------- | --------------------------------- |
| 1   | Random Forest | n=100, depth=5                    |
| 2   | Random Forest | n=200, depth=10                   |
| 3   | Random Forest | n=150, depth=8, max_features=log2 |
| 4   | XGBoost       | lr=0.1, n=100, depth=6            |

---

### 📊 Hasil Eksperimen

| Model       | Accuracy | F1-Macro   | Precision | Recall |
| ----------- | -------- | ---------- | --------- | ------ |
| RF Baseline | 0.9833   | 0.7460     | 0.7422    | 0.75   |
| RF Deep     | 0.9917   | **0.9147** | 0.996     | 0.875  |
| RF Log2     | 0.9917   | 0.9147     | 0.996     | 0.875  |
| XGBoost     | 0.9917   | 0.9147     | 0.996     | 0.875  |

---

### 🏆 Model Terbaik

* **Model:** Random Forest (Deep)
* **Parameter:** n_estimators=200, max_depth=10
* **F1-Macro:** 0.9147
* **Run ID:** `073969d794804d27b3a144c5cf01f79d`

**Alasan pemilihan:**

* F1-score tertinggi (stabil di semua kelas)
* Recall tinggi untuk kelas kritis
* Lebih stabil dibanding baseline

---

### 📈 Monitoring di MLflow

MLflow menyediakan:

* Perbandingan antar run
* Visualisasi metrik
* Tracking parameter
* Penyimpanan model

Untuk melihat:

```bash
mlflow ui
```

---

### 💾 Version Control

Tambahkan MLflow ke `.gitignore`:

```bash
echo "mlruns/" >> .gitignore
```

---

### 🔀 Git Workflow (LK-06)

```bash
git add src/models/train.py
git add .gitignore

git commit -m "feat: add MLflow experiment tracking (LK-06)"

git checkout -b feat/model-training
git push origin feat/model-training
```

---


## 🤖 CI/CD Pipeline — Tutorial

Pipeline berjalan otomatis via GitHub Actions setiap kali ada push ke `main`
atau setiap hari jam **08.00 WIB** (01.00 UTC).

[![CI/CD Status](https://github.com/anazahra/MLOps-AirQualityPrediction/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/anazahra/MLOps-AirQualityPrediction/actions/workflows/mlops_pipeline.yml)

### Alur Pipeline

```
Push / PR ke main
      │
      ▼
┌──────────────────────────┐
│  JOB 1: CI – Validation  │  ← Berjalan di SEMUA push & PR
│  • Install dependencies  │
│  • Run pytest tests/     │
└───────────┬──────────────┘
            │ (jika semua test PASSED)
            ▼
┌──────────────────────────┐
│  JOB 2: CD – Pipeline    │  ← Berjalan HANYA di branch main
│  • Run ingest_data.py    │
│  • Run preprocess.py     │
│  • Upload artifact       │
└──────────────────────────┘
```

### Trigger yang Aktif

| Trigger | Kapan Berjalan |
|---------|----------------|
| `push` ke `main` / `feat/**` | Setiap kali ada commit baru |
| `pull_request` ke `main` | Saat PR dibuka atau diperbarui |
| `schedule` cron `0 1 * * *` | Setiap hari jam 08.00 WIB otomatis |
| `workflow_dispatch` | Manual dari tab Actions di GitHub |

### Menjalankan Workflow Manual

1. Buka tab **Actions** di repositori GitHub
2. Klik **MLOps AQI Pipeline – CI/CD** di sidebar kiri
3. Klik tombol **Run workflow** → pilih branch `main` → **Run workflow**

### Menjalankan Unit Tests Lokal

```bash
# Install pytest jika belum ada
pip install pytest

# Jalankan semua tests
pytest tests/ -v
```

### Lokasi File Workflow

```
.github/workflows/mlops_pipeline.yml
```

## 🔗 Link Repositori

- **GitHub:** https://github.com/anazahra/MLOps-AirQualityPrediction
- **Actions (CI/CD):** https://github.com/anazahra/MLOps-AirQualityPrediction/actions
- **Video Referensi Tutorial:** https://www.youtube.com/watch?v=ciqWMIf7Pz0

## 👤 Kontributor
**Ana Zahratul Firdausi** - 235150201111049 

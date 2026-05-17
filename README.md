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

## 🏷️ Model Registry & Versioning — LK-07

### 📌 Deskripsi

Pada tahap ini, model terbaik dari LK-06 dikelola siklus hidupnya menggunakan
**MLflow Model Registry**. Model didaftarkan secara resmi, diberi versi,
dan ditransisikan melewati tahap Staging hingga Production. Metadata model
juga disinkronisasi menggunakan DVC untuk menjaga data lineage.

---

### 🏆 Model Aktif (Production)

| Komponen | Detail |
|----------|--------|
| Nama Model | AQI_RandomForest |
| Versi Aktif | v1 |
| Stage | **Production** |
| Accuracy | 0.9917 |
| F1-Macro | 0.9147 |
| Precision | 0.996 |
| Recall | 0.875 |
| Run ID | `ef1e7282fead45f9ae8cc4baaf2b4506` |

**Alasan pemilihan v1 sebagai Production:**
Model RF Deep (n_estimators=200, max_depth=10) dipilih karena menghasilkan
nilai accuracy dan F1-macro tertinggi di antara semua run LK-06. Parameter
yang lebih dalam memungkinkan model mempelajari pola data kualitas udara
yang kompleks secara lebih akurat dan stabil.

---

### 📋 Riwayat Versi Model

| Versi | n_estimators | max_depth | Accuracy | F1-Macro | Stage |
|-------|-------------|-----------|----------|----------|-------|
| v1 | 200 | 10 | 0.9917 | 0.9147 | Production |
| v2 | 300 | 12 | 0.9917 | 0.9147 | None |

---

### 📂 Struktur Tambahan

```
src/models/
├── train.py               # Training eksperimen LK-06
├── register_model.py      # Registrasi & versioning model
└── verify_inference.py    # Verifikasi model Production bisa dipanggil

models/
├── model_production.yaml      # Metadata model aktif (tracked DVC)
└── model_production.yaml.dvc  # Pointer DVC
```

---

### ▶️ 1. Registrasi Model Versi Baru (v2)

Model terbaik dari LK-06 otomatis terdaftar saat training. Untuk mendaftarkan
versi baru (v2) dengan parameter berbeda:

```bash
python src/models/register_model.py
```

**Output:** Daftar semua versi model yang terdaftar di MLflow Model Registry

---

### 🔀 2. Transisi Stage Model

Memindahkan model melewati MLflow UI:

```bash
mlflow ui
# Buka browser → tab Models → AQI_RandomForest → Version 1
# Klik Stage → Transition to Staging → Transition to Production
```

| Stage | Deskripsi |
|-------|-----------|
| None | Model baru terdaftar, belum divalidasi |
| Staging | Model dalam pengujian pre-production |
| Production | Model aktif digunakan untuk inferensi |

---

### ✅ 3. Verifikasi Inferensi

Membuktikan model Production dapat dipanggil secara programatik:

```bash
python src/models/verify_inference.py
```

**Output yang diharapkan:**

```
=======================================================
HASIL VERIFIKASI INFERENSI
=======================================================
Model URI   : models:/AQI_RandomForest/Production
Input PM2.5 : 12.5 ug/m3
Prediksi    : Baik
Status      : INFERENSI BERHASIL
=======================================================
```

Load model langsung di kode:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model('models:/AQI_RandomForest/Production')
prediction = model.predict(data)  
```

---

### 📦 4. Metadata Model (DVC)

File metadata model Production di-track dengan DVC untuk menjaga data lineage
antara versi dataset dan versi model yang dihasilkan:

```bash
# Lihat metadata model yang aktif
cat models/model_production.yaml

# Cek status DVC
dvc status
```

| File | Lokasi | Deskripsi |
|------|--------|-----------|
| model_production.yaml | models/ | Metadata model Production aktif |
| model_production.yaml.dvc | models/ | Pointer DVC (di-commit ke Git) |

---

### 💾 Version Control

Tambahkan mlruns ke `.gitignore` jika belum ada:

```bash
echo "mlruns/" >> .gitignore
```

---

---

## 🤖 Otomatisasi End-to-End Pipeline — LK-08

### 📌 Deskripsi

Pada tahap ini, pipeline MLOps dikembangkan menjadi sistem otomatis end-to-end menggunakan **GitHub Actions**. Setiap push ke repository akan memicu tiga job secara berurutan: **Testing → Training → Registry**.

---

### 1. Membuat Branch Baru

```bash
git checkout main
git pull origin main
git checkout -b feat/mlops-automation
```

---

### 2. Konfigurasi Pemicu (Trigger)

#### 2.1 Membuat Struktur File Workflow

```bash
# Membuat file workflow kosong
touch .github/workflows/mlops-automation.yaml

# Verifikasi struktur
ls -la .github/workflows/
```

#### 2.2 Struktur Dasar YAML

```yaml
name: MLOps Pipeline Automation
on:
  push:
    branches: [ 'main' ]
  pull_request:
    branches: [ 'main' ]
```

**Output:** File `.github/workflows/mlops-automation.yaml` terbuat

---

### 3. Job 1: Automated Testing

#### 3.1 Konfigurasi Job Testing

```yaml
jobs:
  testing:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout kode
        uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependensi
        run: pip install -r requirements.txt
      - name: Jalankan unit test
        run: pytest tests/ -v
```

#### 3.2 Verifikasi Test File (lokal)

```bash
pytest tests/ -v
```

**Output:** Seluruh unit test lulus sebelum pipeline dipicu

---

### 4. Job 2: Automated Training

#### 4.1 Konfigurasi Job Training

```yaml
  training:
    runs-on: ubuntu-latest
    needs: testing        # Hanya jalan jika testing LULUS
    steps:
      - name: Checkout kode
        uses: actions/checkout@v4
      - name: Setup Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install dvc
      - name: Pull dataset dari DVC
        env:
          DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          dvc remote modify dagshub auth basic
          dvc remote modify dagshub user $DAGSHUB_USERNAME
          dvc remote modify dagshub password $DAGSHUB_TOKEN
          dvc pull
      - name: Jalankan training model
        env:
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
        run: python src/models/train.py
      - name: Simpan mlruns sebagai artifact
        uses: actions/upload-artifact@v4
        with:
          name: mlruns-artifact-${{ github.run_number }}
          path: mlruns/
          retention-days: 7
```

#### 4.2 Konfigurasi GitHub Secrets

Tambahkan di `Settings → Secrets and variables → Actions`:

| Secret | Keterangan |
|--------|------------|
| `DAGSHUB_USERNAME` | Username DagsHub untuk DVC pull |
| `DAGSHUB_TOKEN` | Token autentikasi DagsHub |
| `OPENWEATHER_API_KEY` | API key OpenWeatherMap |

---

### 5. Job 3: Model Evaluation & Auto-Registry

#### 5.1 Membuat Skrip Evaluasi dan Promosi

Buat file `src/evaluate_and_promote.py`


---

### 6. Simulasi Trigger & Push

#### 6.1 Push Pertama (Deploy Workflow)

```bash
git add .github/workflows/mlops-automation.yaml
git add src/evaluate_and_promote.py
git add src/__init__.py
git add tests/test_pipeline.py
git commit -m "feat: add mlops automation pipeline with testing, training, registry"
git push origin feat/mlops-automation
```

#### 6.2 Trigger Test (Simulasi Perubahan Kode)

```bash
sed -i 's/# RUN 1: Random Forest - Baseline/# RUN 1: Random Forest - Baseline  [trigger test]/' src/models/train.py
grep "trigger test" src/models/train.py
git add src/models/train.py
git commit -m "chore: minor update to trigger automated pipeline (LK-08 simulation)"
git push origin feat/mlops-automation
```


### Alur Pipeline Lengkap

```
Push / PR ke main
      │
      ▼
┌─────────────────────────┐
│  JOB 1: testing         │  ← pytest tests/ -v
│  (selalu berjalan)      │
└──────────┬──────────────┘
           │ (jika PASS)
           ▼
┌─────────────────────────┐
│  JOB 2: training        │  ← python src/models/train.py
│  needs: testing         │    + dvc pull dataset
└──────────┬──────────────┘
           │ (jika SELESAI)
           ▼
┌─────────────────────────┐
│  JOB 3: registry        │  ← python src/evaluate_and_promote.py
│  needs: training        │    → promosi ke MLflow Staging
└─────────────────────────┘
```

### Trigger yang Aktif

| Trigger | Kapan Berjalan |
|---------|----------------|
| `push` ke `main` / `feat/**` | Setiap kali ada commit baru |
| `pull_request` ke `main` | Saat PR dibuka atau diperbarui |

---


## 🐳 Orkestrasi Layanan ML — LK-09

### Arsitektur Sistem
```
┌─────────────────────────────────────────────────────┐
│              Docker Compose — ml-network            │
│                                                     │
│  ┌─────────────────┐    ┌──────────────────────┐   │
│  │  api-service    │───▶│   mlflow-server      │   │
│  │  (port 8000)    │    │   (port 5000)        │   │
│  └─────────────────┘    └──────────┬───────────┘   │
│                                    │                │
│                         ┌──────────▼───────────┐   │
│                         │  Volume: mlflow-data  │   │
│                         └──────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Prasyarat
- Docker & Docker Compose terinstall
- File `.env` berisi `OPENWEATHER_API_KEY`

### Cara Menjalankan Seluruh Sistem
```bash
# 1. Clone repositori
git clone https://github.com/anazahra/MLOps-AirQualityPrediction.git
cd MLOps-AirQualityPrediction

# 2. Buat file .env
cp .env.example .env
# Edit .env dan isi OPENWEATHER_API_KEY dengan API key asli

# 3. Jalankan seluruh sistem dengan SATU perintah
docker compose up -d

# 4. Cek status
docker compose ps

# 5. Akses layanan
# MLflow UI     : http://localhost:5000
# API Inferensi : http://localhost:8000
```

### Menghentikan Sistem
```bash
docker compose down           # Hentikan (data tetap tersimpan)
docker compose down -v        # Hentikan + hapus semua volume
```


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

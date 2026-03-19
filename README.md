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
  - Akses langsung tanpa registrasi: https://data.bmkg.go.id/api/
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
cp .env.example .env  # isi OPENWEATHER_API_KEY di .env
```
 
## 🌳 Branching Strategy (GitHub Flow)
| Branch | Tujuan |
|--------|--------|
| `main` | Kode production-ready, hanya dari Pull Request |
| `feat/initial-eda` | EDA awal data AQI *(sudah di-merge)* |
| `feat/data-pipeline` | Pipeline ingestion & preprocessing (mendatang) |
| `feat/model-training` | Training Random Forest + MLflow (mendatang) |
 
## 👤 Kontributor
**Ana Zahratul Firdausi** - 235150201111049 
## 📄 Lisensi
MIT License - lihat [LICENSE](LICENSE)

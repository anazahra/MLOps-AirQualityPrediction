"""
AQI Data Ingestion Script
=========================
Mengambil data kualitas udara dari OpenWeatherMap Air Pollution API
dengan data yang diambil adalah 10 kota besar Indonesia secara otomatis dan non-destruktif
 
Cara menjalankan:
    python src/ingest_data.py
 
Output:
    data/raw/data_YYYYMMDD_HHMMSS.csv
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konfigurasi API
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Daftar kota
CITIES = [
    {"name": "Jakarta",   "lat": -6.2088,  "lon": 106.8456},
    {"name": "Surabaya",  "lat": -7.2575,  "lon": 112.7521},
    {"name": "Bandung",   "lat": -6.9175,  "lon": 107.6191},
    {"name": "Medan",     "lat": 3.5952,   "lon": 98.6722},
    {"name": "Semarang",  "lat": -6.9932,  "lon": 110.4203},
    {"name": "Makassar",  "lat": -5.1477,  "lon": 119.4327},
    {"name": "Palembang", "lat": -2.9761,  "lon": 104.7754},
    {"name": "Tangerang", "lat": -6.1702,  "lon": 106.6402},
    {"name": "Depok",     "lat": -6.4025,  "lon": 106.7942},
    {"name": "Bekasi",    "lat": -6.2383,  "lon": 106.9756},
]


def fetch_aqi_data(city: dict) -> dict | None:
    """Mengambil data AQI dari OpenWeatherMap Air Pollution API"""
    params = {
        "lat": city["lat"],
        "lon": city["lon"],
        "appid": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        components = data["list"][0]["components"]
        pm25 = components.get("pm2_5")
        aqi_num, aqi_cat = classify_aqi(pm25)
        
        return {
            "city": city["name"],
            "latitude": city["lat"],
            "longitude": city["lon"],
            "timestamp": datetime.utcnow().isoformat(),
            "pm2_5": pm25,
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "co": components.get("co"),
            "so2": components.get("so2"),
            "nh3": components.get("nh3"),
            "aqi": aqi_num,           
            "aqi_category": aqi_cat
        }

    except requests.RequestException as e:
        logger.error(f"Gagal mengambil data untuk {city['name']}: {e}")
        return None


def classify_aqi(pm25: float) -> tuple[int, str]:
    """Mengklasifikasikan kategori AQI berdasarkan PM2.5 (standar BMKG)"""
    if pm25 is None:
        return 0, "Unknown"
    
    if pm25 <= 15:
        return 1, "Baik"
    elif pm25 <= 65:
        return 2, "Sedang"
    elif pm25 <= 150:
        return 3, "Tidak Sehat"
    else:
        return 4, "Berbahaya"
    
def save_to_csv(records: list, output_dir: str = "data/raw") -> str:
    """Menyimpan records ke file CSV dengan timestamp (non-destruktif)"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)

    logger.info(f"Data disimpan ke: {filepath} ({len(df)} records)")
    return filepath


def main():
    """Pipeline utama ingestion data AQI"""
    if not API_KEY:
        logger.error("API key tidak ditemukan !")
        logger.error(
            "Pastikan OPENWEATHE_API_KEY sudah diset di Codespaces Secrets")
        return

    logger.info("=== Memulai data ingestion AQI ===")
    logger.info(f"Mengambil data untuk {len(CITIES)} KOTA ... ")
    records = []
    for city in CITIES:
        logger.info(f"Fetching: {city['name']}...")
        data = fetch_aqi_data(city)
        if data:
            records.append(data)
            logger.info(f" ✅ {city['name']}: PM2.5={data['pm2_5']:.2f} µg/m³, "f"AQI={data['aqi']} ({data['aqi_category']})")
        else:
            logger.warning(f" ❌ {city['name']}: Gagal diambil, dilewati")

    if not records:
        logger.error("Tidak ada data yang berhasil diambil!")
        return

    filepath = save_to_csv(records)
    logger.info("=== Data ingestion selesai ===")
    logger.info(f"Berhasil: {len(records)}/{len(CITIES)} kota")
    logger.info(f"File: {filepath}")

    # Preview data
    df = pd.read_csv(filepath)
    print("\nPreview data: ")
    print(df[["city", "pm2_5", "aqi", "aqi_category"]].round(2).to_string(index=False))

if __name__ == "__main__":
    main()
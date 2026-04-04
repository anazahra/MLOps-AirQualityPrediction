"""
AQI Data Preprocessing Script
==============================
Membersihkan dan memproses data mentah AQI untuk training model.
Pipeline: Load → Clean → Feature Engineering → Save
 
Cara menjalankan:
    python src/preprocess.py
 
Input:
    data/raw/data_*.csv (otomatis detect file terbaru)
 
Output:
    data/processed/features_YYYYMMDD.csv
"""

import os
import logging
import math
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_raw_file(raw_dir: str = "data/raw") -> str:
    """Mencari file CSV terbaru di data/raw/"""
    pattern = os.path.join(raw_dir, "data_*.csv")
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"Tidak ada file data_*.csv di {raw_dir}")

    latest_file = max(files, key=os.path.getctime)
    logger.info(f"File terbaru ditemukan: {os.path.basename(latest_file)}")
    return latest_file


def load_data() -> pd.DataFrame:
    """Memuat data mentah AQI terbaru"""
    latest_file = find_latest_raw_file()
    df = pd.read_csv(latest_file)
    logger.info(f"Data dimuat: {len(df)} records, {len(df.columns)} kolom")
    logger.info(f"Kolom: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data: handle missing values & outlier"""
    logger.info("=== Proses cleaning ===")
    original_len = len(df)

    # 1. Pastikan numeric columns
    numeric_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            missing = df[col].isnull().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    f"  {col}: {missing} missing → median {median_val:.2f}")

    # 2. Remove outlier PM2.5 > 500
    outlier_mask = df["pm2_5"] > 500
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        logger.warning(f"  PM2.5: {outlier_count} outlier (>500) dihapus")
        df = df[~outlier_mask]

    # 3. Drop duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"])
    logger.info(f"  Duplikat: {before_dedup - len(df)} records dihapus")

    logger.info(f"Cleaning selesai: {original_len} → {len(df)} records")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering untuk model ML"""
    logger.info("=== Feature Engineering ===")

    # 1. Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    logger.info("  ✅ Fitur waktu (hour, weekend, month)")

    # 2. Cyclical encoding jam
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).round(6)
    logger.info("  ✅ Cyclical encoding jam")

    # 3. Lag features PM2.5 per kota
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    for lag in [1, 2, 3]:
        df[f"pm25_lag_{lag}"] = df.groupby("city")["pm2_5"].shift(lag)
        df[f"pm25_lag_{lag}"] = df[f"pm25_lag_{lag}"].fillna(df["pm2_5"])
    logger.info("  ✅ Lag features PM2.5 (1,2,3)")

    # 4. Rolling statistics
    df["pm25_rolling_mean_6h"] = (
        df.groupby("city")["pm2_5"].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        ).round(4)
    )
    df["pm25_rolling_std_6h"] = (
        df.groupby("city")["pm2_5"].transform(
            lambda x: x.rolling(window=6, min_periods=1).std()
        ).fillna(0).round(4)
    )
    logger.info("  ✅ Rolling stats PM2.5 (6 jam)")

    # 5. Composite pollution index
    polutan_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
    available_cols = [col for col in polutan_cols if col in df.columns]
    df["pollution_index"] = df[available_cols].mean(axis=1).round(4)
    logger.info("  ✅ Pollution index (mean semua polutan)")

    # 6. Z-score scaling (optional untuk model)
    scale_cols = ["pm2_5", "pm10", "no2", "o3", "co"]
    for col in scale_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f"{col}_zscore"] = ((df[col] - mean_val) / std_val).round(6)

    logger.info("  ✅ Z-score scaling polutan")
    return df


def save_processed(df: pd.DataFrame, output_dir: str = "data/processed") -> str:
    """Simpan data processed"""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"features_{date_str}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"✅ Data processed: {filepath} ({df.shape})")
    return filepath


def main():
    """Pipeline preprocessing lengkap"""
    logger.info("🚀 === MULAI PREPROCESSING AQI ===")

    # 1. Load latest raw data
    df = load_data()

    # 2. Clean
    df_clean = clean_data(df)

    # 3. Feature engineering
    df_features = engineer_features(df_clean)

    # 4. Save
    output_path = save_processed(df_features)

    logger.info("🎉 === PREPROCESSING SELESAI ===")
    logger.info(f"📊 Shape akhir: {df_features.shape}")

    # Ringkasan dataset
    print("\n" + "="*50)
    print("📈 Ringkasan Dataset")
    print("="*50)
    print(f"Total records: {len(df_features):,}")
    print(f"Total kolom: {len(df_features.columns)}")
    print(f"\nDistribusi AQI Kategori:")
    print(df_features['aqi_category'].value_counts())
    print(f"\nPM2.5 Stats:")
    print(df_features['pm2_5'].describe().round(2))
    print(f"\nFitur baru yang ditambahkan:")
    new_cols = [col for col in df_features.columns if any(
        x in col for x in ['lag_', 'rolling_', 'hour_', 'zscore'])]
    print(
        f"  {len(new_cols)} fitur: {new_cols[:5]}{'...' if len(new_cols)>5 else ''}")


if __name__ == "__main__":
    main()

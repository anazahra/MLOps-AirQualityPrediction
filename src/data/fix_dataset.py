import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load full dataset
df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
logger.info(f"✅ Loaded {len(df)} rows from dataset_combined_kedua.csv")

# Minimal cleaning (SESUAIKAN kolom)
required_cols = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'aqi_category']  # Adjust!
df = df[required_cols].dropna()
logger.info(f"✅ After cleaning: {len(df)} rows")

# Save IMMEDIATELY
output_path = 'data/processed/features_20260404_FIXED.csv'
df.to_csv(output_path, index=False)
logger.info(f"💾 Saved {len(df)} rows to {output_path}")

print(f"✅ FIXED! Gunakan: data/processed/features_20260404_FIXED.csv")
print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
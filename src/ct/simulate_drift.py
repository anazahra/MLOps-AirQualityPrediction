"""
Buat dataset shifted untuk simulasi data drift.
Data digeser sedemikian rupa sehingga distribusi PM2.5 dan polutan
lainnya berbeda signifikan dari data training (mensimulasikan kondisi polusi meningkat drastis)
"""
import pandas as pd
import numpy as np
import os

def simulate_drift(input_path='data/raw/dataset_combined_kedua.csv', output_path='data/raw/dataset_shifted.csv', shift_factor=2.0, n_samples=100):

    print('=== Simulasi Data Drift ===')
    print(f'Input  : {input_path}')
    print(f'Output : {output_path}')
    print(f'Shift  : x{shift_factor} pada semua polutan')
    print(f'Sampel : {n_samples} rows')

    df = pd.read_csv(input_path)

    # Ambil sampel acak
    df_shifted = df.sample(n=min(n_samples, len(df)), random_state=42).copy()

    # Inject data sintetis untuk range PM2.5 yang kosong di data asli
    # (PM2.5 32-42 setelah x2.0 -> 64-84 -> masuk Baru_TidakSehat)
    n_inject = 10
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi']
    coords = {
        'Jakarta':   (-6.2088,  106.8456),
        'Surabaya':  (-7.2575,  112.7521),
        'Bandung':   (-6.9175,  107.6191),
        'Medan':     ( 3.5952,   98.6722),
        'Semarang':  (-6.9932,  110.4203),
        'Makassar':  (-5.1477,  119.4327),
        'Palembang': (-2.9761,  104.7754),
        'Tangerang': (-6.1702,  106.6402),
        'Depok':     (-6.4025,  106.7942),
        'Bekasi':    (-6.2383,  106.9756),
    }

    df_inject = pd.DataFrame({
        'city':      cities,
        'latitude':  [coords[c][0] for c in cities],
        'longitude': [coords[c][1] for c in cities],
        'timestamp': [pd.Timestamp.now().isoformat()] * n_inject,
        'aqi':       [3] * n_inject,
        'pm2_5':     np.linspace(32, 42, n_inject),
        'pm10':      np.linspace(38, 50, n_inject),
        'no2':       [5.0] * n_inject,
        'o3':        [80.0] * n_inject,
        'co':        [400.0] * n_inject,
        'so2':       [3.0] * n_inject,
        'nh3':       [1.5] * n_inject,
        'aqi_category': ['Sedang'] * n_inject
    })
    df_shifted = pd.concat([df_shifted, df_inject], ignore_index=True)

    # Geser nilai polutan
    pollutant_cols = ['pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3']
    for col in pollutant_cols:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted[col] * shift_factor

    # Update aqi dan aqi_category menggunakan aturan baru dari clustering
    def pm25_to_aqi_category(pm25):
        if pm25 < 36.885:
            return 1, 'Baru_Baik'
        elif pm25 < 63.045:
            return 2, 'Baru_Sedang'
        elif pm25 < 85.3:
            return 3, 'Baru_TidakSehat'
        else:
            return 4, 'Baru_Berbahaya'

    aqi_vals = df_shifted['pm2_5'].apply(lambda x: pm25_to_aqi_category(x)[0])
    cat_vals = df_shifted['pm2_5'].apply(lambda x: pm25_to_aqi_category(x)[1])
    df_shifted['aqi'] = aqi_vals
    df_shifted['aqi_category'] = cat_vals

    # Update timestamp
    from datetime import datetime, timedelta
    base_time = datetime.now()
    df_shifted['timestamp'] = [
        (base_time + timedelta(minutes=i)).isoformat()
        for i in range(len(df_shifted))
    ]

    df_shifted.to_csv(output_path, index=False)

    print('\nDistribusi aqi_category setelah shift:')
    print(df_shifted['aqi_category'].value_counts().to_string())
    print(f'\nFile disimpan: {output_path}')
    print(f'Total rows   : {len(df_shifted)}')
    print('\nPerbandingan PM2.5:')
    print(f'  Asli   : mean={df["pm2_5"].mean():.3f}, std={df["pm2_5"].std():.3f}')
    print(f'  Shifted: mean={df_shifted["pm2_5"].mean():.3f}, std={df_shifted["pm2_5"].std():.3f}')

if __name__ == '__main__':
    simulate_drift()
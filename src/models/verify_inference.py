"""Verifikasi Inferensi Model Production"""
import mlflow.pyfunc, pandas as pd, numpy as np, logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_inference():
    logger.info('=== VERIFIKASI INFERENSI MODEL PRODUCTION ===')
    MODEL_URI = 'models:/AQI_RandomForest/Production'
    logger.info(f'Memuat model dari: {MODEL_URI}')
    model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
    logger.info('Model berhasil dimuat!')

    # Data sampel lengkap - 25 kolom sesuai fitur training
    pm25 = 12.5
    hour = 10

    sample = pd.DataFrame([{
        # Polutan dasar
        'pm2_5': pm25, 'pm10': 20.3, 'no2': 1.5,
        'o3': 50.2, 'co': 210.0, 'so2': 1.8, 'nh3': 0.5,
        'aqi': 1,
        # Fitur waktu
        'hour': hour, 'day_of_week': 1, 'month': 4,
        'is_weekend': 0,
        # Cyclical encoding
        'hour_sin': round(np.sin(2 * np.pi * hour / 24), 6),
        'hour_cos': round(np.cos(2 * np.pi * hour / 24), 6),
        # Lag features
        'pm25_lag_1': 11.0, 'pm25_lag_2': 10.5, 'pm25_lag_3': 12.0,
        # Rolling stats
        'pm25_rolling_mean_6h': 11.5, 'pm25_rolling_std_6h': 0.5,
        # Pollution index
        'pollution_index': round((pm25+20.3+1.5+50.2+210.0+1.8)/6, 4),
        # Z-score (nilai 0 = rata-rata, aman untuk sampel)
        'pm2_5_zscore': 0.0, 'pm10_zscore': 0.0,
        'no2_zscore': 0.0, 'o3_zscore': 0.0, 'co_zscore': 0.0,
    }])

    logger.info(f'Input data ({len(sample.columns)} kolom):\n{sample.to_string()}')
    prediction = model.predict(sample)

    label_map = {0: 'Baik', 1: 'Sedang', 2: 'Tidak Sehat', 3: 'Berbahaya'}
    predicted_label = label_map.get(int(prediction[0]), str(prediction[0]))

    print('\n' + '='*55)
    print('HASIL VERIFIKASI INFERENSI')
    print('='*55)
    print(f'Model URI   : {MODEL_URI}')
    print(f'Input PM2.5 : {pm25} ug/m3')
    print(f'Prediksi    : {predicted_label}')
    print(f'Status      : INFERENSI BERHASIL')
    print('='*55)

if __name__ == '__main__':
    try:
        verify_inference()
    except Exception as e:
        print(f'Error: {e}')
        print('Pastikan model sudah berstatus Production!')

"""
Metrics Exporter untuk Prometheus
=========================================
Menyediakan endpoint /metrics yang dapat di-scrape Prometheus.
Mencatat metrik operasional: latensi, throughput, prediksi.
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import time

# Definisi Metrik 

# Jumlah total request prediksi yang diterima
PREDICTION_COUNTER = Counter(
    'aqi_prediction_requests_total',
    'Total jumlah request prediksi AQI',
    ['status']  # label: success / error
)

# Histogram latensi inferensi (dalam detik)
INFERENCE_LATENCY = Histogram(
    'aqi_inference_latency_seconds',
    'Latensi inferensi model AQI dalam detik',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Distribusi nilai prediksi (untuk mendeteksi data drift)
PREDICTION_VALUE = Histogram(
    'aqi_prediction_value',
    'Distribusi kelas prediksi AQI (0=Baik, 1=Sedang, 2=Tidak Sehat, 3=Berbahaya)',
    buckets=[0, 1, 2, 3, 4]
)

# Gauge untuk PM2.5 input (bisa berubah naik turun)
PM25_INPUT_GAUGE = Gauge(
    'aqi_input_pm25',
    'Nilai PM2.5 terakhir yang diterima sebagai input prediksi'
)

# Context Manager untuk mengukur latensi

def track_inference(func):
    """Decorator untuk mengukur latensi dan mencatat prediksi."""
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            INFERENCE_LATENCY.observe(duration)
            PREDICTION_COUNTER.labels(status='success').inc()
            return result
        except Exception as e:
            PREDICTION_COUNTER.labels(status='error').inc()
            raise e
    return wrapper


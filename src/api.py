"""
Update: api.py dengan endpoint /metrics untuk Prometheus
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
import mlflow.pyfunc
import pandas as pd
import os, time, logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.metrics_exporter import (
    PREDICTION_COUNTER, INFERENCE_LATENCY,
    PREDICTION_VALUE, PM25_INPUT_GAUGE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='AQI Prediction API', version='1.1.0-LK11')

# Load model saat startup
MODEL_URI = os.getenv('MODEL_URI', 'models:/AQI_RandomForest/Production')
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
mlflow.set_tracking_uri(MLFLOW_URI)

model = None

@app.on_event('startup')
async def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info(f'Model loaded from {MODEL_URI}')
    except Exception as e:
        logger.warning(f'Model load failed: {e}. API running without model.')

@app.get('/')
def root():
    return {'status': 'ok', 'service': 'AQI Prediction API', 'version': '1.1.0-LK11'}

@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}

# Endpoint Metrik Prometheus 
@app.get('/metrics')
def metrics():
    """Endpoint yang di-scrape oleh Prometheus setiap 15 detik."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Endpoint Prediksi
@app.post('/predict')
def predict(request: dict):
    if model is None:
        PREDICTION_COUNTER.labels(status='error').inc()
        return {'error': 'Model not loaded'}
    
    start = time.time()
    try:
        data = pd.DataFrame([request])
        
        # Catat nilai PM2.5 untuk monitoring drift
        if 'pm2_5' in request:
            PM25_INPUT_GAUGE.set(request['pm2_5'])
        
        prediction = model.predict(data)
        pred_class = int(prediction[0])
        
        # Catat metrik
        duration = time.time() - start
        INFERENCE_LATENCY.observe(duration)
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_VALUE.observe(pred_class)
        
        label_map = {0: 'Baik', 1: 'Sedang', 2: 'Tidak Sehat', 3: 'Berbahaya'}
        return {
            'prediction': pred_class,
            'category': label_map.get(pred_class, 'Unknown'),
            'latency_ms': round(duration * 1000, 2)
        }
    except Exception as e:
        PREDICTION_COUNTER.labels(status='error').inc()
        return {'error': str(e)}


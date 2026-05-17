FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode proyek
COPY . .

# Expose port API
EXPOSE 8000

# Jalankan FastAPI dengan uvicorn
CMD ["python", "-c", "
import mlflow, os;
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'));
print('API service ready. MLFLOW_TRACKING_URI:', os.getenv('MLFLOW_TRACKING_URI'));
import time; time.sleep(3600)"]
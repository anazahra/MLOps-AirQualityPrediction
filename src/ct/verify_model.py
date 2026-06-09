import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = 'AQI_RandomForest'

def load_production_model():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == 'Production']
    
    if not prod:
        raise Exception("Tidak ada model Production!")
    
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model v{prod[0].version} berhasil diload")
    return model

def categorize(pm2_5):
    if pm2_5 < 36.885:   return 'Baru_Baik'
    elif pm2_5 < 63.045: return 'Baru_Sedang'
    elif pm2_5 < 85.3:   return 'Baru_TidakSehat'
    else:                return 'Baru_Berbahaya'

def generate_unseen_data(n=300):
    np.random.seed(99)  # Seed berbeda dari training (42)
    
    data = {
        'aqi':   np.random.normal(80, 40, n).clip(0, 500),
        'pm2_5': np.random.normal(55, 25, n).clip(0, 500),  # range mencakup semua kelas
        'pm10':  np.random.normal(60, 30, n).clip(0, 600),
        'no2':   np.random.normal(20, 10, n).clip(0, 200),
        'o3':    np.random.normal(40, 15, n).clip(0, 200),
        'co':    np.random.normal(0.8, 0.3, n).clip(0, 10),
        'so2':   np.random.normal(10, 5, n).clip(0, 100),
        'nh3':   np.random.normal(5, 2, n).clip(0, 50),
    }
    
    df = pd.DataFrame(data)
    df['aqi_category'] = df['pm2_5'].apply(categorize)
    
    print(f"\nDistribusi label data unseen:")
    print(df['aqi_category'].value_counts())
    
    return df

def verify():
    print("=" * 55)
    print("Verifikasi Model dengan Data Unseen")
    print("=" * 55)
    
    model = load_production_model()
    print(f"\nFitur model: {list(model.feature_names_in_)}")
    
    df = generate_unseen_data(n=300)
    X = df.drop(columns=['aqi_category'])
    X = X[model.feature_names_in_]  # reorder otomatis sesuai model
    
    y_true = df['aqi_category'].astype(str)
    y_pred_encoded = model.predict(X)

    model_classes = model.classes_  # tetap seperti punyamu
    y_pred = [str(model_classes[i]) for i in y_pred_encoded]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\nHasil pada Data Unseen (n=300):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Macro : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("=" * 55)
    if acc >= 0.90:
        print("✅ Model generalize dengan baik!")
    elif acc >= 0.75:
        print("⚠️  Performa turun, tapi masih wajar")
    else:
        print("❌ Model kemungkinan overfit! Perlu investigasi")

if __name__ == '__main__':
    verify()
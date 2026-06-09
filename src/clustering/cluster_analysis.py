"""
Clustering Analysis 
====================================================
Tujuan: Menemukan pola alami dalam data AQI menggunakan K-Means,
lalu membuat aturan klasifikasi baru berdasarkan hasil cluster.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load dataset"""
    df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Kolom: {list(df.columns)}")
    return df

def prepare_features_for_clustering(df):
    """Pilih fitur polutan utama untuk clustering"""
    # Semua kolom polutan numerik
    # TIDAK dimasukkan: city, latitude, longitude, timestamp (identitas, bukan kondisi udara)
    # TIDAK dimasukkan: aqi_category (ini label/target -> data leakage kalau dipakai sebagai fitur)
    feature_cols = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'co', 'so2', 'nh3']
    # Filter hanya kolom yang ada di dataset (jaga-jaga)
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(df[available].median())
    print(f"Fitur untuk clustering: {available}")
    
    # Normalisasi Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X, available, scaler

def find_optimal_k(X_scaled, k_range=range(2, 7)):
    """
    Mencari jumlah cluster optimal dengan Elbow Method dan Silhouette Score.
    Menargetkan k=4 agar selaras dengan 4 kategori BMKG, tapi tetap cek apakah data memang memiliki 4 kelompok alami.
    """
    inertias = []
    silhouettes = []
    
    print("\n=== Mencari k Optimal ===")
    print(f"{'k':>4} {'Inertia':>12} {'Silhouette':>12}")
    print("-" * 32)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"{k:>4} {inertia:>12.2f} {sil:>12.4f}")
    
    # Pilih k dengan silhouette tertinggi
    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"\nK terbaik berdasarkan Silhouette Score: k={best_k}")
    print(f"(Catatan: mencoba k=4 untuk perbandingan dengan BMKG)")
    return best_k, inertias, silhouettes

def run_kmeans(X_scaled, X_original, df, n_clusters=4):
    """Menjalankan K-Means dengan k=4 dan analisis setiap cluster yang terbentuk"""
    print(f"\n=== K-Means Clustering (k={n_clusters}) ===")
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    
    # Tambahkan label cluster ke dataframe asli
    df_result = df.copy()
    df_result['cluster_id'] = cluster_labels
    
    # Analisis setiap cluster
    print("\n=== Profil Setiap Cluster ===")
    feature_cols = X_original.columns.tolist()
    
    cluster_profiles = {}
    for cid in sorted(df_result['cluster_id'].unique()):
        subset = df_result[df_result['cluster_id'] == cid]
        profile = {
            'count': len(subset),
            'pm2_5_mean': subset['pm2_5'].mean() if 'pm2_5' in subset else 0,
            'pm10_mean': subset['pm10'].mean() if 'pm10' in subset else 0,
            'aqi_category_dist': subset['aqi_category'].value_counts().to_dict() \
                                  if 'aqi_category' in subset else {}
        }
        cluster_profiles[cid] = profile
        
        print(f"\nCluster {cid} ({profile['count']} data):")
        for col in feature_cols:
            if col in subset.columns:
                print(f"  {col:>10}: mean={subset[col].mean():.3f}, "
                      f"std={subset[col].std():.3f}")
        if 'aqi_category' in subset.columns:
            print(f"  Distribusi label BMKG: ")
            for cat, cnt in profile['aqi_category_dist'].items():
                pct = cnt / profile['count'] * 100
                print(f"    {cat}: {cnt} ({pct:.1f}%)")
    
    return df_result, km, cluster_profiles

def create_new_rules(df_with_clusters, cluster_profiles):
    """
    Membuat aturan klasifikasi baru berdasarkan profil cluster.
    Aturan = threshold pada fitur PM2.5 (polutan paling representatif).
    """
    print("\n=== Membuat Aturan Baru dari Cluster ===")
    
    # Urutkan cluster berdasarkan rata-rata PM2.5 (ascending)
    cluster_pm25 = {}
    for cid, profile in cluster_profiles.items():
        cluster_pm25[cid] = profile['pm2_5_mean']
    
    sorted_clusters = sorted(cluster_pm25.items(), key=lambda x: x[1])
    print("Urutan cluster dari PM2.5 terendah ke tertinggi:")
    for rank, (cid, mean_val) in enumerate(sorted_clusters):
        print(f"  Rank {rank+1}: Cluster {cid} (PM2.5 mean = {mean_val:.3f})")
    
    # Buat mapping cluster -> kategori baru
    category_names = ['Baru_Baik', 'Baru_Sedang', 'Baru_TidakSehat', 'Baru_Berbahaya']
    cluster_to_category = {}
    for rank, (cid, _) in enumerate(sorted_clusters):
        if rank < len(category_names):
            cluster_to_category[cid] = category_names[rank]
    
    df_with_clusters['new_category'] = df_with_clusters['cluster_id'].map(cluster_to_category)
    
    # Hitung threshold PM2.5 untuk aturan baru
    # Threshold = rata-rata antara batas atas cluster i dan batas bawah cluster i+1
    thresholds = {}
    for i in range(len(sorted_clusters) - 1):
        cid_lower = sorted_clusters[i][0]
        cid_upper = sorted_clusters[i+1][0]
        subset_lower = df_with_clusters[df_with_clusters['cluster_id'] == cid_lower]
        subset_upper = df_with_clusters[df_with_clusters['cluster_id'] == cid_upper]
        if 'pm2_5' in subset_lower.columns:
            max_lower = subset_lower['pm2_5'].max()
            min_upper = subset_upper['pm2_5'].min()
            threshold = (max_lower + min_upper) / 2
            thresholds[f"threshold_{i+1}"] = round(threshold, 3)
    
    print("\n=== Aturan  Baru ===")
    print(f"  PM2.5 < {thresholds.get('threshold_1', '?')} -> {category_names[0]}")
    print(f"  PM2.5 {thresholds.get('threshold_1', '?')} - {thresholds.get('threshold_2', '?')} -> {category_names[1]}")
    print(f"  PM2.5 {thresholds.get('threshold_2', '?')} - {thresholds.get('threshold_3', '?')} -> {category_names[2]}")
    print(f"  PM2.5 >= {thresholds.get('threshold_3', '?')} -> {category_names[3]}")
    print("\n=== Aturan BMKG sebagai Referensi ===")
    print("  PM2.5 0-15.5   -> Baik")
    print("  PM2.5 15.5-55  -> Sedang")
    print("  PM2.5 55-150   -> Tidak Sehat")
    print("  PM2.5 >150     -> Berbahaya")
    
    return df_with_clusters, thresholds, cluster_to_category

def compare_with_bmkg(df_with_clusters):
    """
    Membandingkan label BMKG dengan label dari aturan baru.
    Hitung seberapa besar perbedaannya
    """
    print("\n=== Perbandingan Label BMKG vs Aturan Baru ===")
    
    if 'aqi_category' not in df_with_clusters.columns:
        print("Kolom aqi_category tidak ditemukan, skip perbandingan")
        return
    
    # Normalisasi nama kategori BMKG agar bisa dibandingkan
    category_mapping = {
        'Baik': 'Baru_Baik',
        'Sedang': 'Baru_Sedang',
        'Tidak Sehat': 'Baru_TidakSehat',
        'Berbahaya': 'Baru_Berbahaya'
    }
    df_with_clusters['bmkg_normalized'] = df_with_clusters['aqi_category'].map(
        lambda x: next((v for k, v in category_mapping.items() if k in str(x)), x)
    )
    
    total = len(df_with_clusters)
    match = (df_with_clusters['bmkg_normalized'] == df_with_clusters['new_category']).sum()
    pct_match = match / total * 100
    
    print(f"Total data: {total}")
    print(f"Label sama antara BMKG dan Aturan Baru: {match} ({pct_match:.1f}%)")
    print(f"Label berbeda: {total - match} ({100 - pct_match:.1f}%)")
    print("\nKesimpulan:")
    if pct_match < 80:
        print("  Data riil BERBEDA signifikan dari aturan BMKG.")
        print("  -> Aturan baru dari clustering lebih merepresentasikan pola data!")
    else:
        print("  Data riil SESUAI dengan aturan BMKG (perbedaan < 20%).")
        print("  -> BMKG sudah cukup baik, aturan baru sebagai validasi.")
    
    # Crosstab
    print("\nCrosstab BMKG vs Aturan Baru:")
    ct = pd.crosstab(
        df_with_clusters['aqi_category'],
        df_with_clusters['new_category'],
        margins=True
    )
    print(ct.to_string())
    return pct_match

def train_with_new_labels(df_with_clusters, feature_cols):
    """
    Training model Random Forest menggunakan label dari clustering.
    Lalu membandingkan dengan model yang menggunakan label BMKG.
    """
    print("\n=== Traing Model dengan Label Aturan Baru ===")
    mlflow.set_experiment('AQI_Clustering_NewRules')
    
    # --- Drop baris yang tidak punya label baru ---
    df_clean = df_with_clusters.dropna(subset=['new_category'])
    available_features = [c for c in feature_cols if c in df_clean.columns]
    
    X = df_clean[available_features].fillna(0)
    y_new = df_clean['new_category']  # label aturan baru
    y_bmkg = df_clean['aqi_category'] if 'aqi_category' in df_clean.columns else None
    
    X_train, X_test, y_new_train, y_new_test = train_test_split(
        X, y_new, test_size=0.2, random_state=42
    )
    
    # Model 1: Training dengan label aturan baru 
    with mlflow.start_run(run_name="RF_NewRules_Label"):
        mlflow.log_param("label_source", "clustering_new_rules")
        mlflow.log_param("n_clusters", 4)
        
        clf_new = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         random_state=42, n_jobs=-1)
        clf_new.fit(X_train, y_new_train)
        y_pred_new = clf_new.predict(X_test)
        
        acc_new = accuracy_score(y_new_test, y_pred_new)
        f1_new = f1_score(y_new_test, y_pred_new, average='macro', zero_division=0)
        
        mlflow.log_metric('accuracy', round(acc_new, 4))
        mlflow.log_metric('f1_macro', round(f1_new, 4))
        mlflow.sklearn.log_model(clf_new, 'model',
                                  registered_model_name='AQI_NewRules_RF')
        
        print(f"\nModel dengan Label Aturan Baru:")
        print(f"  Accuracy : {acc_new:.4f}")
        print(f"  F1-Macro : {f1_new:.4f}")
        print(f"  Report:\n{classification_report(y_new_test, y_pred_new, zero_division=0)}")
    
    # Model 2 (jika ada label BMKG): Training dengan label BMKG 
    if y_bmkg is not None:
        y_bmkg_train = y_bmkg.loc[X_train.index]
        y_bmkg_test = y_bmkg.loc[X_test.index]
        
        with mlflow.start_run(run_name="RF_BMKG_Label"):
            mlflow.log_param("label_source", "bmkg_rules")
            
            clf_bmkg = RandomForestClassifier(n_estimators=200, max_depth=10,
                                              random_state=42, n_jobs=-1)
            clf_bmkg.fit(X_train, y_bmkg_train)
            y_pred_bmkg = clf_bmkg.predict(X_test)
            
            acc_bmkg = accuracy_score(y_bmkg_test, y_pred_bmkg)
            f1_bmkg = f1_score(y_bmkg_test, y_pred_bmkg, average='macro', zero_division=0)
            
            mlflow.log_metric('accuracy', round(acc_bmkg, 4))
            mlflow.log_metric('f1_macro', round(f1_bmkg, 4))
            
            print(f"\nModel dengan Label BMKG:")
            print(f"  Accuracy : {acc_bmkg:.4f}")
            print(f"  F1-Macro : {f1_bmkg:.4f}")
        
        print("\n=== PERBANDINGAN FINAL ===")
        print(f"{'Model':<30} {'Accuracy':>10} {'F1-Macro':>10} {'Label Sumber':>15}")
        print("-" * 70)
        print(f"{'RF + Label Aturan Baru':<30} {acc_new:>10.4f} {f1_new:>10.4f} {'Clustering':>15}")
        print(f"{'RF + Label BMKG':<30} {acc_bmkg:>10.4f} {f1_bmkg:>10.4f} {'BMKG Rules':>15}")
    
    return clf_new, X_test, y_new_test

def test_new_rules(clf_new, X_test, y_new_test, thresholds):
    """Testing akhir model dengan label aturan baru dan test menggunakan rule-based (threshold)"""
    print("\n=== Testing Aturan Baru ===")
    
    # Prediksi menggunakan model RF yang sudah dilatih dengan aturan baru
    y_pred = clf_new.predict(X_test)
    
    acc = accuracy_score(y_new_test, y_pred)
    f1 = f1_score(y_new_test, y_pred, average='macro', zero_division=0)
    
    print(f"Hasil Testing pada Data Test ({len(X_test)} sampel):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Macro : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_new_test, y_pred, zero_division=0))
    
    # Contoh prediksi pada data baru (simulasi)
    print("\n=== Simulasi Prediksi Data Baru ===")
    sample_data = pd.DataFrame([{
        'aqi': 2, 'pm2_5': 12.5, 'pm10': 20.0, 'no2': 1.5,
        'o3': 50.0, 'co': 200.0, 'so2': 1.0, 'nh3': 0.5
    }])
    # Filter kolom sesuai fitur model
    sample_cols = [c for c in X_test.columns if c in sample_data.columns]
    sample_filtered = sample_data[sample_cols]
    
    pred = clf_new.predict(sample_filtered)
    print(f"  Input PM2.5=12.5 -> Prediksi: {pred[0]}")
    print(f"  (Berdasarkan aturan BMKG: PM2.5=12.5 -> Baik)")
    
    return acc, f1

def main():
    print("=" * 60)
    print("Clustering & Aturan Baru")
    print("=" * 60)
    
    # 1. Load data
    df = load_data()
    
    # 2. Persiapan fitur
    X_scaled, X_original, feature_cols, scaler = prepare_features_for_clustering(df)
    
    # 3. Cari k optimal
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)
    
    # 4. Jalankan K-Means dengan k=4
    df_result, km_model, cluster_profiles = run_kmeans(X_scaled, X_original, df, n_clusters=4)
    
    # 5. Buat aturan baru
    df_result, thresholds, cluster_to_cat = create_new_rules(df_result, cluster_profiles)
    
    # 6. Bandingkan dengan BMKG
    pct_match = compare_with_bmkg(df_result)
    
    # 7. Training dengan label baru
    clf_new, X_test, y_new_test = train_with_new_labels(df_result, feature_cols)
    
    # 8. Testing
    acc, f1 = test_new_rules(clf_new, X_test, y_new_test, thresholds)
    
    # 9. Simpan hasil ke CSV untuk dokumentasi
    output_path = 'data/processed/clustering_results.csv'
    df_result.to_csv(output_path, index=False)
    print(f"\nHasil clustering disimpan ke: {output_path}")
    
    print("\n" + "=" * 60)
    print("Selesai! Ringkasan:")
    print(f"  - Jumlah cluster: 4")
    print(f"  - Keselarasan dengan BMKG: {pct_match:.1f}%")
    print(f"  - Accuracy model aturan baru: {acc:.4f}")
    print(f"  - F1-Macro model aturan baru: {f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
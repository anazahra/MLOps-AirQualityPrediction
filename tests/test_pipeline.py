"""Unit Tests untuk MLOps AQI Pipeline"""
import pandas as pd
import glob
import os
import pytest


class TestIngestion:
    """Test untuk proses data ingestion"""

    def test_dataset_file_exists(self):
        """File dataset_combined.csv harus ada"""
        assert os.path.exists('data/raw/dataset_combined.csv'), \
            'dataset_combined.csv tidak ditemukan!'

    def test_required_columns_exist(self):
        """Kolom wajib harus ada di dataset"""
        df = pd.read_csv('data/raw/dataset_combined.csv')
        required = ['city', 'timestamp', 'pm2_5', 'aqi', 'aqi_category']
        for col in required:
            assert col in df.columns, f'Kolom wajib tidak ada: {col}'

    def test_no_empty_dataset(self):
        """Dataset tidak boleh kosong"""
        df = pd.read_csv('data/raw/dataset_combined.csv')
        assert len(df) > 0, 'Dataset kosong!'

    def test_aqi_category_valid_values(self):
        """Nilai aqi_category harus sesuai standar BMKG"""
        df = pd.read_csv('data/raw/dataset_combined.csv')
        valid = {'Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'}
        invalid = set(df['aqi_category'].unique()) - valid
        assert len(invalid) == 0, f'Kategori AQI tidak valid: {invalid}'

    def test_ten_cities_covered(self):
        """Harus ada data untuk 10 kota"""
        df = pd.read_csv('data/raw/dataset_combined.csv')
        n_cities = df['city'].nunique()
        assert n_cities >= 10, f'Hanya ada {n_cities} kota, expected >= 10'


class TestPreprocessing:
    """Test untuk proses preprocessing"""

    def test_processed_file_exists(self):
        """File features_*.csv harus ada di data/processed/"""
        files = glob.glob('data/processed/features_*.csv')
        assert len(files) > 0, 'Tidak ada file processed di data/processed/'

    def test_processed_has_30_columns(self):
        """Output preprocessing harus punya 30 kolom"""
        files = glob.glob('data/processed/features_*.csv')
        df = pd.read_csv(files[-1])
        assert len(df.columns) == 30, \
            f'Expected 30 kolom, got {len(df.columns)}'

    def test_no_null_values(self):
        """Tidak boleh ada nilai null di processed data"""
        files = glob.glob('data/processed/features_*.csv')
        df = pd.read_csv(files[-1])
        null_count = df.isnull().sum().sum()
        assert null_count == 0, f'Ada {null_count} nilai null di processed data!'


class TestDVCVersioning:
    """Test untuk DVC file versioning"""

    def test_dvc_file_exists(self):
        """File .dvc harus ada sebagai bukti versioning"""
        assert os.path.exists('data/raw/dataset_combined.csv.dvc'), \
            'File .dvc tidak ditemukan — apakah DVC sudah diinisialisasi?'

    def test_dvc_config_exists(self):
        """Konfigurasi DVC harus ada"""
        assert os.path.exists('.dvc/config'), \
            '.dvc/config tidak ditemukan — jalankan dvc init terlebih dahulu'

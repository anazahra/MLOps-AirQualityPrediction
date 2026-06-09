"""Unit Tests untuk MLOps AQI Pipeline"""
import pandas as pd
import glob
import os
import pytest

# File data hanya ada di Codespaces, tidak di GitHub Actions runner
DATA_AVAILABLE = os.path.exists('data/raw/dataset_combined_kedua.csv')
PROCESSED_AVAILABLE = len(glob.glob('data/processed/features_*.csv')) > 0


class TestIngestion:

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="File data tidak tersedia di CI")
    def test_dataset_file_exists(self):
        assert os.path.exists('data/raw/dataset_combined_kedua.csv')

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="File data tidak tersedia di CI")
    def test_required_columns_exist(self):
        df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
        required = ['city', 'timestamp', 'pm2_5', 'aqi', 'aqi_category']
        for col in required:
            assert col in df.columns

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="File data tidak tersedia di CI")
    def test_no_empty_dataset(self):
        df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
        assert len(df) > 0

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="File data tidak tersedia di CI")
    def test_aqi_category_valid_values(self):
        df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
        valid = {'Baik', 'Sedang', 'Tidak Sehat', 'Berbahaya'}
        invalid = set(df['aqi_category'].unique()) - valid
        assert len(invalid) == 0, f'Kategori AQI tidak valid: {invalid}'

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="File data tidak tersedia di CI")
    def test_ten_cities_covered(self):
        df = pd.read_csv('data/raw/dataset_combined_kedua.csv')
        assert df['city'].nunique() >= 10

class TestPreprocessing:

    @pytest.mark.skipif(not PROCESSED_AVAILABLE, reason="File processed tidak tersedia di CI")
    def test_processed_file_exists(self):
        files = glob.glob('data/processed/features_*.csv')
        assert len(files) > 0

    @pytest.mark.skipif(not PROCESSED_AVAILABLE, reason="File processed tidak tersedia di CI")
    def test_processed_has_30_columns(self):
        files = glob.glob('data/processed/features_*.csv')
        df = pd.read_csv(files[-1])
        assert len(df.columns) >= 20

    @pytest.mark.skipif(not PROCESSED_AVAILABLE, reason="File processed tidak tersedia di CI")
    def test_no_null_values(self):
        files = glob.glob('data/processed/features_*.csv')
        df = pd.read_csv(files[-1])
        assert df.isnull().sum().sum() == 0

class TestDVCVersioning:

    def test_dvc_file_exists(self):
        """Selalu jalan: file .dvc ada di Git"""
        assert os.path.exists('data/raw/dataset_combined_kedua.csv.dvc')

    def test_dvc_config_exists(self):
        """Selalu jalan: .dvc/config ada di Git"""
        assert os.path.exists('.dvc/config')

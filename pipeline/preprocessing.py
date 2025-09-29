# preprocessing.py (Final Version)

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.feature_columns = [
            'tpep_pickup_datetime',
            'tpep_dropoff_datetime', 
            'passenger_count',
            'trip_distance',
            'RatecodeID',
            'PULocationID',
            'DOLocationID',
            'payment_type',
            'extra'
        ]
        self.target_column = 'total_amount'
        self.fitted = False
        self.feature_names = None
    
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        df = df.copy()
        
        # --- TODO: UPDATE THIS LINE WITH THE FORMAT YOU FOUND ---
        # After running debug_dates.py, put the correct format string here.
        # For example: datetime_format = '%m/%d/%Y %H:%M:%S'
        datetime_format = '%m/%d/%Y %I:%M:%S %p' # <-- REPLACE THIS GUESS WITH YOUR ACTUAL FORMAT

        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=datetime_format, errors='coerce')
        
        # ... (the rest of the file is the same as the last version) ...
        
        if 'tpep_pickup_datetime' in df.columns:
            df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
            df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
            df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
            df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
        
        if 'tpep_dropoff_datetime' in df.columns:
            df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour
            df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day
            df['dropoff_month'] = df['tpep_dropoff_datetime'].dt.month
            df['dropoff_weekday'] = df['tpep_dropoff_datetime'].dt.weekday
        
        if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['tpep_dropoff_datetime']) and \
               pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
                df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
                df['trip_duration'] = df['trip_duration'].clip(0, 1440)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        df = df.copy()
        
        cols_to_convert = [
            'total_amount', 'trip_distance', 'passenger_count', 'extra',
            'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type'
        ]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=self.feature_columns + [self.target_column])
        
        if 'total_amount' in df.columns:
            df = df[df['total_amount'].between(0, 1000)]
        if 'trip_distance' in df.columns:
            df = df[df['trip_distance'].between(0, 100)]
        if 'passenger_count' in df.columns:
            df = df[df['passenger_count'].between(1, 8)].astype({'passenger_count': 'int'})
        if 'extra' in df.columns:
            df = df[df['extra'] >= 0]
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        df = df.copy()
        df = self.extract_datetime_features(df)
        
        if 'trip_duration' in df.columns:
             df = df.dropna(subset=['trip_duration'])
             df = df[df['trip_duration'].between(0, 300)]
        
        numeric_features = [
            'passenger_count', 'trip_distance', 'RatecodeID',
            'PULocationID', 'DOLocationID', 'payment_type', 'extra'
        ]
        time_features = [
            'pickup_hour', 'pickup_day', 'pickup_month', 'pickup_weekday',
            'dropoff_hour', 'dropoff_day', 'dropoff_month', 'dropoff_weekday',
            'trip_duration'
        ]
        
        feature_list = numeric_features + time_features
        existing_features = [f for f in feature_list if f in df.columns]
        features_df = df[existing_features].copy()
        features_df = features_df.fillna(features_df.median())
        
        return features_df

    def fit(self, df: pd.DataFrame):
        """仅拟合(学习)数据的缩放参数"""
        df_clean = self.clean_data(df)
        if len(df_clean) == 0:
            raise ValueError("No data remaining after cleaning for fitting.")
        
        X_df = self.prepare_features(df_clean)
        if len(X_df) == 0:
             raise ValueError("No data remaining after preparing features for fitting.")

        y = df_clean.loc[X_df.index][self.target_column].values
        
        self.scaler_features.fit(X_df)
        self.scaler_target.fit(y.reshape(-1, 1))
        
        self.fitted = True
        self.feature_names = X_df.columns.tolist()
        
        print(f"Preprocessor fitted. Feature dimension: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """使用已学习的参数转换新数据"""
        if not self.fitted:
            raise ValueError("Must call fit() before transforming data.")
        
        df_clean = self.clean_data(df)
        if len(df_clean) == 0:
            return np.array([]).reshape(0, len(self.feature_names)), np.array([])
        
        X_df = self.prepare_features(df_clean)
        if len(X_df) == 0:
            return np.array([]).reshape(0, len(self.feature_names)), np.array([])
        
        y = df_clean.loc[X_df.index][self.target_column].values
        X_df = X_df.reindex(columns=self.feature_names, fill_value=0) 
        
        X = self.scaler_features.transform(X_df)
        y_scaled = self.scaler_target.transform(y.reshape(-1, 1)).flatten()
        
        return X, y_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并转换训练数据 (方便单进程使用)"""
        self.fit(df)
        return self.transform(df)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """将标准化的目标变量转换回原始尺度"""
        return self.scaler_target.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        if not self.fitted:
            raise ValueError("Must call fit() first.")
        return self.feature_names.copy()

def train_test_split_local(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在本地(单个进程)数据上执行训练和测试集划分
    """
    if X.shape[0] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    np.random.seed(seed)
    total_rows = X.shape[0]
    all_indices = np.arange(total_rows)
    np.random.shuffle(all_indices)
    
    test_size = int(total_rows * test_ratio)
    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    return X_train, y_train, X_test, y_test
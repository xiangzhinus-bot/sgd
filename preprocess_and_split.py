import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler, train_test_split


#转化数据类型的函数
def optimize_dtypes(df):
    """
    在不改变数值的前提下优化数据类型以减少内存占用
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # 处理整数列
        if col_type == 'int64':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # 处理浮点数列
        elif col_type == 'float64':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # 处理对象列（字符串）
        elif col_type == 'object':
            if df_optimized[col].nunique() / len(df_optimized[col]) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


#对数据集进行预处理和分割的函数
def preprocess_and_split(df):

    # 提取指定的列
    selected_columns = [
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime', 
        'passenger_count',
        'trip_distance',
        'RatecodeID',
        'PULocationID',
        'DOLocationID',
        'payment_type',
        'extra', 
        'total_amount'
    ]
    # 选择这些列
    extracted_df = df[selected_columns]

    #处理时间类型数据
    time_format = '%m/%d/%Y %I:%M:%S %p'
    extracted_df['tpep_pickup_datetime'] = pd.to_datetime(extracted_df['tpep_pickup_datetime'], format=time_format)
    extracted_df['tpep_dropoff_datetime'] = pd.to_datetime(extracted_df['tpep_dropoff_datetime'], format=time_format)
    extracted_df['time_on_taxi'] = extracted_df['tpep_dropoff_datetime'] - extracted_df['tpep_pickup_datetime']
    extracted_df = extracted_df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis = 1)
    
    # 优化数据类型
    df_optimized = optimize_dtypes(extracted_df)

    #处理缺失值(删除存在缺失值的行)
    na_columns = df_optimized.columns[df_optimized.isna().any()].tolist()
    print(na_columns)
    df_dropped = df_optimized.dropna()

    #归一化数据（minmax）
    scaler = MinMaxScaler()
    df_normalized_array = scaler.fit_transform(df_dropped)
    # 将结果转换回 DataFrame
    df_normalized = pd.DataFrame(df_normalized_array, columns=df_dropped.columns)

    #处理完毕的数据集
    final_df = df_normalized

    #分割训练集和测试集
    X = final_df.drop(columns=['total_amount'])
    y = final_df.loc[:,'total_amount']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test

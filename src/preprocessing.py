import numpy as np
import pandas as pd

def remove_extreme_outliers(df, features=['V14','V12','V10'], class_col='Class', class_value=1, iqr_multiplier=1.5):
    """
    Loại bỏ extreme outliers cho các feature quan trọng (default V14, V12, V10)
    
    Parameters:
    - df: DataFrame input
    - features: list các feature cần loại bỏ outlier
    - class_col: tên cột class
    - class_value: giá trị class để tính IQR (thường fraud=1)
    - iqr_multiplier: hệ số nhân với IQR để xác định threshold
    
    Returns:
    - df_clean: DataFrame sau khi loại bỏ outlier
    """
    
    df_clean = df.copy()
    
    for feature in features:
        # Lấy giá trị feature cho fraud (hoặc class_value)
        vals = df_clean[feature].loc[df_clean[class_col] == class_value].values
        
        # Tính IQR
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q75 - q25
        cut_off = iqr * iqr_multiplier
        lower, upper = q25 - cut_off, q75 + cut_off
        
        # Tìm outlier
        outliers = [x for x in vals if x < lower or x > upper]
        print(f'Feature {feature}: {len(outliers)} outliers detected')
        
        # Loại bỏ outlier
        df_clean = df_clean.drop(df_clean[(df_clean[feature] < lower) | (df_clean[feature] > upper)].index)
        print(f'Number of instances after removing {feature} outliers: {len(df_clean)}')
        print('----'*10)
    
    return df_clean


from sklearn.preprocessing import RobustScaler

def scale_amount_time_train_test(train_df, test_df, scaler_type='robust'):
    """
    Scale Amount và Time, fit trên train, transform train + test
    """
    if scaler_type=='robust':
        scaler_amount = RobustScaler()
        scaler_time = RobustScaler()
    elif scaler_type=='standard':
        from sklearn.preprocessing import StandardScaler
        scaler_amount = StandardScaler()
        scaler_time = StandardScaler()
    else:
        raise ValueError("scaler_type phải là 'robust' hoặc 'standard'")
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Fit scaler trên train
    train_df['scaled_amount'] = scaler_amount.fit_transform(train_df[['Amount']])
    train_df['scaled_time']   = scaler_time.fit_transform(train_df[['Time']])
    
    # Transform test
    test_df['scaled_amount'] = scaler_amount.transform(test_df[['Amount']])
    test_df['scaled_time']   = scaler_time.transform(test_df[['Time']])
    
    # Drop cột gốc
    train_df.drop(['Amount','Time'], axis=1, inplace=True)
    test_df.drop(['Amount','Time'], axis=1, inplace=True)
    
    return train_df, test_df

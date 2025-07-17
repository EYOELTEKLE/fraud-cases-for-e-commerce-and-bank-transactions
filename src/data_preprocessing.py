import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from ydata_profiling import ProfileReport


# Utility functions for IP conversion
def ip_to_int(ip_str):
    if pd.isnull(ip_str) or not isinstance(ip_str, str):
        return 0  # or -1, or np.nan, depending on your downstream logic
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return 0
# Load data
def load_data(fraud_path, ip_country_path):
    fraud = pd.read_csv(fraud_path)
    ip_country = pd.read_csv(ip_country_path)
    profileFraud = ProfileReport(fraud)
    profileIpcountry = ProfileReport(ip_country)
    print(fraud.describe(include='all').T)
    print(ip_country.describe(include='all').T)
    profileFraud.to_file("report_fraud.html")
    profileIpcountry.to_file("report_ipcountry.html")
    return fraud, ip_country

# Handle missing values and clean data
def clean_fraud_data(df):
    df = df.drop_duplicates()
    # Impute or drop missing values
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    # Correct data types
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
    return df

# Merge with geolocation
def merge_with_geolocation(fraud, ip_country):
    fraud['ip_int'] = fraud['ip_address'].apply(ip_to_int)
    ip_country['lower_int'] = ip_country['lower_bound_ip_address'].apply(ip_to_int)
    ip_country['upper_int'] = ip_country['upper_bound_ip_address'].apply(ip_to_int)
    # Merge using interval join
    ip_country = ip_country.sort_values('lower_int')
    fraud = fraud.sort_values('ip_int')
    fraud['country'] = None
    idx = 0
    for i, row in fraud.iterrows():
        while idx < len(ip_country) and row['ip_int'] > ip_country.iloc[idx]['upper_int']:
            idx += 1
        if idx < len(ip_country) and ip_country.iloc[idx]['lower_int'] <= row['ip_int'] <= ip_country.iloc[idx]['upper_int']:
            fraud.at[i, 'country'] = ip_country.iloc[idx]['country']
    return fraud

# Feature engineering
def feature_engineering(df):
    df['transaction_hour'] = df['purchase_time'].dt.hour
    df['transaction_dayofweek'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0
    # Transaction frequency and velocity
    df = df.sort_values(['user_id', 'purchase_time'])
    # Compute transactions in the last 24 hours for each transaction
    df['transactions_last_24h'] = 0
    for user_id, group in df.groupby('user_id'):
        times = group['purchase_time']
        counts = []
        for i, t in enumerate(times):
            window_start = t - pd.Timedelta(hours=24)
            count = times[(times > window_start) & (times <= t)].count()
            counts.append(count)
        df.loc[group.index, 'transactions_last_24h'] = counts
    return df

# Encode categorical features
def encode_categorical(df, categorical_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df

# Normalize/scale features
def scale_features(df, columns, scaler_type='standard'):
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Handle class imbalance
def resample_data(X, y, method='smote'):
    if method == 'smote':
        sampler = SMOTE(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# EDA (placeholder, to be expanded in notebooks or scripts)
def eda_summary(df):
    print(df.describe(include='all'))
    print(df['class'].value_counts())

if __name__ == "__main__":
    # Example usage
    fraud, ip_country = load_data(
        "data/raw/Fraud_Data.csv",
        "data/raw/IpAddress_to_Country.csv"
    )
    fraud = clean_fraud_data(fraud)
    fraud = merge_with_geolocation(fraud, ip_country)
    fraud = feature_engineering(fraud)
    categorical_cols = ['source', 'browser', 'sex', 'country']
    fraud = encode_categorical(fraud, categorical_cols)
    scale_cols = ['purchase_value', 'age', 'transaction_hour', 'transaction_dayofweek', 'time_since_signup', 'transactions_last_24h']
    fraud = scale_features(fraud, scale_cols, scaler_type='standard')
    X = fraud.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_int'])
    y = fraud['class']
    X_res, y_res = resample_data(X, y, method='smote')
    eda_summary(fraud)

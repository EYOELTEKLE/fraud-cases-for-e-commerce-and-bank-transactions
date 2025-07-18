import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import clean_fraud_data, merge_with_geolocation, feature_engineering, encode_categorical, scale_features

def load_fraud_data():
    fraud = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_country = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    fraud = clean_fraud_data(fraud)
    fraud = merge_with_geolocation(fraud, ip_country)
    fraud = feature_engineering(fraud)
    categorical_cols = ['source', 'browser', 'sex', 'country']
    fraud = encode_categorical(fraud, categorical_cols)
    scale_cols = ['purchase_value', 'age', 'transaction_hour', 'transaction_dayofweek', 'time_since_signup', 'transactions_last_24h']
    fraud = scale_features(fraud, scale_cols, scaler_type='standard')
    X = fraud.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'ip_int'])
    y = fraud['class']
    return X, y

def run_shap_explainability(X, y):
    # Use a subset for SHAP speed if dataset is large
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    # SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    # Summary plot (global feature importance)
    shap.summary_plot(shap_values, X_test, show=True)
    # Force plot (local explanation for first prediction)
    shap.initjs()
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0], matplotlib=True, show=True)

if __name__ == "__main__":
    X, y = load_fraud_data()
    run_shap_explainability(X, y)

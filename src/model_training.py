import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Utility for evaluation

def evaluate_model(y_true, y_pred, y_proba, model_name):
    print(f"\nModel: {model_name}")
    print("F1 Score:", f1_score(y_true, y_pred))
    print("AUC-PR:", average_precision_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {average_precision_score(y_true, y_proba):.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.show()

# Load processed data (Fraud_Data)
def load_fraud_data():
    # Assumes data_preprocessing.py has generated a processed CSV or returns processed df
    # For demonstration, rerun preprocessing pipeline here
    from data_preprocessing import clean_fraud_data, merge_with_geolocation, feature_engineering, encode_categorical, scale_features
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

# Load processed data (creditcard.csv)
def load_creditcard_data():
    df = pd.read_csv("data/raw/creditcard.csv")
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

# Model training and evaluation pipeline
def train_and_evaluate(X, y, model_name="Fraud_Data"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_lr, y_proba_lr, f"Logistic Regression ({model_name})")
    # Ensemble Model (Random Forest)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_rf, y_proba_rf, f"Random Forest ({model_name})")
    # Ensemble Model (XGBoost)
    xgb = XGBClassifier(n_estimators=100, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_xgb, y_proba_xgb, f"XGBoost ({model_name})")

if __name__ == "__main__":
    print("\n--- Fraud_Data.csv Results ---")
    X_fraud, y_fraud = load_fraud_data()
    train_and_evaluate(X_fraud, y_fraud, model_name="Fraud_Data")

    print("\n--- creditcard.csv Results ---")
    X_cc, y_cc = load_creditcard_data()
    train_and_evaluate(X_cc, y_cc, model_name="CreditCard")

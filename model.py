import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_and_preprocess_data(file_path):
    print(f"Loading dataset from: {file_path}")
    dtype_dict = {
        'event_type': 'str', 'requires_road_closure': 'str', 'priority': 'str',
        'corridor': 'str', 'zone': 'str', 'police_station': 'str',
        'veh_type': 'str', 'junction': 'str', 'event_cause': 'str'
    }
    df = pd.read_csv(file_path, dtype=dtype_dict)

    # Datetime parsing
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='ISO8601', utc=True, errors='coerce')
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], format='ISO8601', utc=True, errors='coerce')

    # Target resolution mins
    df['resolution_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
    df = df.dropna(subset=['resolution_mins', 'start_datetime'])
    df = df[df['resolution_mins'] >= 0]
    df['resolution_mins'] = np.clip(df['resolution_mins'].values, a_min=0, a_max=1440)

    # Filter Low (< 30 mins) vs High (> 120 mins)
    df = df[(df['resolution_mins'] < 30) | (df['resolution_mins'] > 120)].copy()
    df['target'] = np.where(df['resolution_mins'] < 30, 0, 1)
    
    print(f"Filtered Shape: {df.shape}")
    return df

def engineer_features(df):
    print("Engineering features...")

    # Datetime features
    df['hour'] = df['start_datetime'].dt.hour.astype('int8')
    df['day_of_week'] = df['start_datetime'].dt.dayofweek.astype('int8')
    df['month'] = df['start_datetime'].dt.month.astype('int8')
    df['is_peak_hour'] = np.where(((df['hour'] >= 8) & (df['hour'] <= 11)) | 
                                  ((df['hour'] >= 17) & (df['hour'] <= 20)), 1, 0).astype('int8')

    # Road closure
    df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.upper().str.strip()
    df['requires_road_closure'] = np.where(df['requires_road_closure'] == 'TRUE', 1, 0).astype('int8')

    # Heavy vehicle check
    df['veh_type'] = df['veh_type'].fillna('unknown').astype(str).str.strip().str.lower()
    return df

def encode_and_split(df):
    print("Encoding categoricals and splitting chronologically...")
    
    # Global Target encoding maps for production compatibility
    global_junction_risk = df.groupby('junction')['target'].mean().to_dict()
    df['junction_risk'] = df['junction'].map(global_junction_risk).fillna(0.5)

    # Label encode textual dimensions globally
    categorical_features = ['event_cause', 'event_type', 'priority', 'corridor', 'zone', 'police_station', 'veh_type', 'junction']
    encoders = {}
    for col in categorical_features:
        df[col] = df[col].fillna('unknown').astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype('int16')
        encoders[col] = le

    # Define final features (removing has_vehicle due to zero importance contribution)
    final_features = [
        'hour', 'day_of_week', 'month', 'event_cause', 'event_type',
        'requires_road_closure', 'priority', 'corridor', 'zone', 
        'police_station', 'veh_type', 'is_peak_hour', 'junction_risk'
    ]

    # Chronological split at 2024-03-01
    split_date = pd.to_datetime('2024-03-01 00:00:00+00', utc=True)
    train_mask = df['start_datetime'] < split_date
    test_mask = df['start_datetime'] >= split_date

    X_train = df.loc[train_mask, final_features]
    y_train = df.loc[train_mask, 'target']
    X_test = df.loc[test_mask, final_features]
    y_test = df.loc[test_mask, 'target']

    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, final_features, encoders, global_junction_risk

def main():
    # Setup paths
    file_path = "c:/Users/HP/Documents/gidlock/theme2.csv"
    if not os.path.exists(file_path):
        file_path = r"C:\Users\HP\Downloads\theme2.csv"
        
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        sys.exit(1)

    # Preprocessing
    df = load_and_preprocess_data(file_path)
    df = engineer_features(df)
    X_train, y_train, X_test, y_test, final_features, encoders, global_junction_risk = encode_and_split(df)

    # Calculate scale_pos_weight
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight_value = num_neg / max(1, num_pos)

    # XGBoost
    print("\n--- Training XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight_value,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(classification_report(y_test, y_pred_xgb))

    # LightGBM
    print("\n--- Training LightGBM ---")
    lgb = LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight_value,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    y_pred_lgb = lgb.predict(X_test)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print(classification_report(y_test, y_pred_lgb))

    # CatBoost
    print("\n--- Training CatBoost ---")
    cat = CatBoostClassifier(
        iterations=200,
        depth=3,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight_value,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cat.fit(X_train, y_train)
    y_pred_cat = cat.predict(X_test)
    print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
    print(classification_report(y_test, y_pred_cat))

    # Voting Classifier Ensemble
    print("\n============================================================")
    print("                    ENSEMBLE EVALUATION REPORT")
    print("============================================================")
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
        voting='soft',
        weights=[0, 5, 1]
    )
    ensemble.fit(X_train, y_train)
    p_ensemble = ensemble.predict_proba(X_test)[:, 1]
    y_pred_ens = (p_ensemble >= 0.55).astype(int)

    print(f"Accuracy Score  : {accuracy_score(y_test, y_pred_ens):.4f}")
    print(f"Macro F1-Score  : {f1_score(y_test, y_pred_ens, average='macro'):.4f}")
    print("\nDetailed Performance Matrix:")
    print(classification_report(y_test, y_pred_ens, target_names=['Low (< 30 mins)', 'High (> 120 mins)']))
    print("Confusion Matrix Layout:")
    print(confusion_matrix(y_test, y_pred_ens))
    print("============================================================")

    # Save production pipeline using VotingClassifier trained on 100% of data
    print("\nTraining final production VotingClassifier ensemble model on 100% of filtered data...")
    prod_ensemble = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                objective='binary:logistic', scale_pos_weight=0.95,
                tree_method='hist', random_state=42, n_jobs=-1
            )),
            ('lgb', LGBMClassifier(
                n_estimators=300, max_depth=7, learning_rate=0.03,
                scale_pos_weight=0.95, random_state=42, n_jobs=-1, verbose=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=200, depth=3, learning_rate=0.01,
                scale_pos_weight=0.95, random_seed=42, verbose=0, thread_count=-1
            ))
        ],
        voting='soft',
        weights=[0, 5, 1]
    )
    
    X_all = df[final_features]
    y_all = df['target']
    prod_ensemble.fit(X_all, y_all)

    # Remove 'junction' from encoders to prevent KeyError inside recommendation_engine.py
    if 'junction' in encoders:
        del encoders['junction']

    print("Saving production model pipeline to 'traffic_impact_production_pipeline.pkl'...")
    artifacts = {
        'model': prod_ensemble,
        'features': final_features,
        'encoders': encoders,
        'junction_risk_map': global_junction_risk,
        'threshold': 0.55
    }
    joblib.dump(artifacts, 'traffic_impact_production_pipeline.pkl')
    print("Successfully generated production pipeline pickle!\n")

if __name__ == "__main__":
    main()

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_hybrid_wips():
    print("[*] Loading extracted features...")
    df = pd.read_csv("wifi_traffic_features.csv", low_memory=False)
    
    # Dropping the deterministic protocol flags and MAC address to force behavioral learning
    X = df.drop(columns=['Target_Class', 'wlan.ta', 'wlan.fc.type', 'wlan.fc.subtype'], errors='ignore')
    y = df['Target_Class']
    
    # Force convert all features to numeric, coerce text/hex to NaN, then fill with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Encode Target_Class 
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("[*] Training Stage 1: XGBoost (Multi-Class Signature Detection)...")
    # Filter out Unknown Attacks so XGBoost never sees the Zero-Days during training
    X_xgb = X[y != 'Unknown_Attack']
    y_xgb = y_encoded[y != 'Unknown_Attack']
    
    X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.3, random_state=42)
    
    # Dynamically generate weights for all classes to fix the 99% Normal traffic imbalance
    print("[*] Computing sample weights for multi-class dataset...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Eval metric changed to 'mlogloss' to support multiple attack classes
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, 
                              use_label_encoder=False, eval_metric='mlogloss')
    
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("\nXGBoost Performance on Specific Attacks:")
    y_pred = xgb_model.predict(X_test)
    
    # Dynamically extract the exact string names for whatever classes are present
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    actual_target_names = le.inverse_transform(unique_labels)
    
    print(classification_report(y_test, y_pred, target_names=actual_target_names))
    
    # --- CONFUSION MATRIX PLOTTING ---
    print("[*] Generating updated Confusion Matrix image...")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actual_target_names, 
                yticklabels=actual_target_names)
                
    # Injects the calculated accuracy directly into the chart title
    plt.title(f'AWID3 WIPS XGBoost Multi-Class Confusion Matrix\n(Accuracy: {acc*100:.2f}%)', pad=15, fontweight='bold')
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    print("[*] Training Stage 2: Isolation Forest (Zero-Day Anomaly Detection)...")
    # Train IF strictly on Normal traffic baseline
    X_normal = X[y == 'Normal']
    iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iforest.fit(X_normal)
    
    print("[*] Saving Hybrid IDS models...")
    joblib.dump(xgb_model, 'xgboost_wips.pkl')
    joblib.dump(iforest, 'iforest_wips.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("[+] Models saved successfully.")

if __name__ == "__main__":
    train_hybrid_wips()
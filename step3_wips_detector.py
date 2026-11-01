import pandas as pd
import joblib
import subprocess
import time
import warnings
import sys

# Suppress sklearn/xgboost warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def run_live_hybrid_wips():
    print("[*] Initializing Hybrid WIPS Engine...")
    
    try:
        xgb_model = joblib.load('xgboost_wips.pkl')
        iforest = joblib.load('iforest_wips.pkl')
        le = joblib.load('label_encoder.pkl')
    except Exception as e:
        print(f"[-] Error loading models: {e}")
        print("[-] Ensure you have run step2_train_model.py first.")
        sys.exit(1)

    target_mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    print(f"[*] Loading simulated live traffic into RAM ({target_mode.upper()} mode)...")
    
    try:
        full_df = pd.read_csv("wifi_traffic_features.csv", low_memory=False)
    except FileNotFoundError:
        print("[-] wifi_traffic_features.csv not found. Run step1_prepare_data.py first.")
        sys.exit(1)
    
    if target_mode == "demo":
        # Force a concentrated mix: 1500 normal frames and 50 guaranteed attacks
        normal_traffic = full_df[full_df['Target_Class'] == 'Normal'].sample(n=1500, random_state=42)
        attack_traffic = full_df[full_df['Target_Class'] != 'Normal'].sample(n=50, random_state=42)
        live_traffic = pd.concat([normal_traffic, attack_traffic]).sample(frac=1, random_state=42)
    else:
        live_traffic = full_df.sample(frac=1, random_state=42)

    print("\n======================================================================")
    print("STEP 3: Real-Time Hybrid WIPS Traffic Monitor (AWID3 Replay)")
    print("======================================================================\n")

    frames_scanned = 0
    quarantined_macs = set()

    for index, row in live_traffic.iterrows():
        mac_address = str(row.get('wlan.ta', 'UNKNOWN'))
        actual_label = row.get('Target_Class', 'Unknown')
        
        # Prepare the frame exactly as it was during training (drop deterministic headers)
        frame_features = row.drop(labels=['Target_Class', 'wlan.ta', 'wlan.fc.type', 'wlan.fc.subtype'], errors='ignore').to_frame().T
        
        # Force numeric conversion and fill NaNs with 0
        for col in frame_features.columns:
            frame_features[col] = pd.to_numeric(frame_features[col], errors='coerce')
        frame_features = frame_features.fillna(0)

        # ---------------------------------------------------------
        # STAGE 1: XGBoost (Known Attack Detection)
        # ---------------------------------------------------------
        xgb_pred_encoded = xgb_model.predict(frame_features)[0]
        xgb_pred_label = le.inverse_transform([xgb_pred_encoded])[0]

        if xgb_pred_label not in ['Normal', 'Unknown_Attack']:
            if mac_address not in quarantined_macs and mac_address != 'UNKNOWN':
                print(f"\n[!] ALERT: {xgb_pred_label} Attack Detected - Malicious behavior profile matched.")
                print(f"    Target MAC: {mac_address}")
                try:
                    subprocess.run(["iptables", "-A", "INPUT", "-m", "mac", "--mac-source", mac_address, "-j", "DROP"], check=True)
                    print(f"[+] WIPS AUTO-PREVENTION: {mac_address} QUARANTINED via iptables.")
                    quarantined_macs.add(mac_address)
                except subprocess.CalledProcessError:
                    print(f"[-] Failed to apply iptables rule for {mac_address}. Are you running as root?")
            continue 

        # ---------------------------------------------------------
        # STAGE 2: Isolation Forest (Zero-Day Detection)
        # ---------------------------------------------------------
        iforest_pred = iforest.predict(frame_features)[0]
        
        if iforest_pred == -1: 
            if mac_address not in quarantined_macs and mac_address != 'UNKNOWN':
                print(f"\n[!] ALERT: ZERO-DAY ANOMALY DETECTED - Statistical deviation from baseline.")
                print(f"    Target MAC: {mac_address}")
                try:
                    subprocess.run(["iptables", "-A", "INPUT", "-m", "mac", "--mac-source", mac_address, "-j", "DROP"], check=True)
                    print(f"[+] WIPS AUTO-PREVENTION: {mac_address} QUARANTINED via iptables.")
                    quarantined_macs.add(mac_address)
                except subprocess.CalledProcessError:
                    print(f"[-] Failed to apply iptables rule for {mac_address}. Are you running as root?")
            continue 

        # ---------------------------------------------------------
        # NORMAL TRAFFIC HANDLING
        # ---------------------------------------------------------
        frames_scanned += 1
        if frames_scanned % 500 == 0:
            print(f"[*] Engine running... {frames_scanned} frames scanned silently (Normal Traffic)")
            
        # Optional micro-sleep to visually simulate live packet streaming during demo
        if target_mode == "demo":
            time.sleep(0.001)

    print("\n[+] Simulation complete. Verify live kernel rules applied to networking stack:")
    try:
        subprocess.run(["iptables", "-L", "INPUT", "-v", "-n"])
    except Exception as e:
        print(f"[-] Could not display iptables: {e}")

if __name__ == "__main__":
    run_live_hybrid_wips()
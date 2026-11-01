import pandas as pd
import glob
import os
import gc

FILE_PREFIXES = {
    "Known_Attack": ["Deauth", "Disas", "RogueAP", "Evil_Twin"],
    "Unknown_Attack": ["Krack", "Kr00k"],
    "Normal": []
}

FEATURES_TO_KEEP = [
    'frame.time_delta', 
    'wlan.fc.type', 
    'wlan.fc.subtype', 
    'wlan.duration',
    'radiotap.dbm_antsignal', 
    'wlan.seq',
    'Label',
    'wlan.ta' 
]

def load_and_label_data():
    output_file = "wifi_traffic_features.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print("[*] Loading AWID3 Data (Multi-Class Streaming Mode)...")
    
    first_file = True
    total_frames = 0
    
    def process_files(prefixes, attack_category):
        nonlocal first_file, total_frames
        for prefix in prefixes:
            path = f"./{prefix}*.csv" 
            for file in glob.glob(path):
                print(f"[*] Processing: {file}")
                try:
                    df = pd.read_csv(file, usecols=lambda c: c in FEATURES_TO_KEEP, low_memory=False)
                    
                    if attack_category == "Known_Attack":
                        df['Target_Class'] = df['Label'] 
                    else:
                        # THE FIX: Ensure background traffic in zero-day folders stays "Normal"
                        df['Target_Class'] = df['Label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Unknown_Attack')
                    
                    df = df.drop(columns=['Label'], errors='ignore').dropna()
                    df.to_csv(output_file, mode='a', header=first_file, index=False)
                    
                    first_file = False
                    frames_added = len(df)
                    total_frames += frames_added
                    print(f"    -> Extracted {frames_added} frames. Clearing from RAM.")
                    
                    del df
                    gc.collect()
                    
                except Exception as e:
                    print(f"    [-] Skipped or error processing {file}: {e}")

    process_files(FILE_PREFIXES["Known_Attack"], "Known_Attack")
    process_files(FILE_PREFIXES["Unknown_Attack"], "Unknown_Attack")

    print(f"\n[+] Extraction complete! Successfully processed {total_frames} total frames.")
    
    if os.path.exists(output_file):
        print("[*] Final Dataset Distribution:")
        final_check = pd.read_csv(output_file, usecols=['Target_Class'])
        print(final_check['Target_Class'].value_counts())

if __name__ == "__main__":
    load_and_label_data()
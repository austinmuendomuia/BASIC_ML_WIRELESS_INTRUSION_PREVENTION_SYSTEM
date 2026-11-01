#!/bin/bash

# Flush old iptables rules for a clean slate before the demo
echo "[*] Flushing existing iptables rules..."
iptables -F

echo "========================================"
echo "       AWID3 HYBRID WIPS ENGINE         "
echo "========================================"
echo "Select execution mode:"
echo "1) Demo Mode (Guarantees attacks for presentation)"
echo "2) Full Mode (Scans entire 500,000+ frame dataset)"
read -p "Enter choice (1 or 2): " mode

if [ "$mode" = "1" ]; then
    echo "[*] Launching in DEMO MODE..."
    ./venv/bin/python3 step3_wips_detector.py demo

elif [ "$mode" = "2" ]; then
    echo "[*] Launching in FULL MODE..."
    ./venv/bin/python3 step3_wips_detector.py full

else
    echo "[-] Invalid selection. Exiting."
    exit 1
fi
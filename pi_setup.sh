#!/bin/bash

set -e

echo "========================================="
echo "Voice Pipeline - Raspberry Pi 5 Setup"
echo "========================================="

echo "[1/5] Updating package list..."
sudo apt update

echo "[2/5] Installing system dependencies..."
sudo apt install -y python3-pip python3-dev portaudio19-dev libportaudio2 libportaudio-common

echo "[3/5] Installing Python dependencies for ARM64..."
pip3 install --upgrade pip setuptools wheel

echo "[4/5] Installing Voice Pipeline requirements..."
pip3 install -r requirements.txt

echo "[5/5] Verifying installation..."
python3 -c "import speech_recognition; import sounddevice; import numpy; import torch; import requests; print('All packages installed successfully!')"

echo "========================================="
echo "Setup complete! Run: python3 main.py"
echo "========================================="
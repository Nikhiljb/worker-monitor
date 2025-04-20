# Worker Activity Monitoring System

## Overview
This is a vision-based AI system that detects if a worker is actively engaged in their task. It runs standalone and logs activity.

✅ Environment Setup for Ultralytics + PyTorch with CUDA
🔧 Step 1: Create a New Virtual Environment

python -m venv yolov-env

📂 Step 2: Activate the Environment
For Windows:

yolov-env\Scripts\activate
For Linux/macOS:

source yolov-env/bin/activate
📦 Step 3: Upgrade pip

pip install --upgrade pip
⚡ Step 4: Install PyTorch with CUDA Support
Visit the PyTorch official installation page for your configuration.

Example (for CUDA 12.2):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
Replace cu122 with your CUDA version. Common options:

CUDA 11.8 → cu118

CUDA 12.1 → cu121

CPU only → cpu

🚀 Step 5: Install Ultralytics (YOLO)

pip install ultralytics
✅ Step 6: Verify CUDA Availability in PyTorch
Create a Python file named check_cuda.py with the following content:

python

import torch

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device Found")
Run the script:

python check_cuda.py
You should see output like this if CUDA is enabled:

yaml

Torch Version: 2.x.x
CUDA Available: True
CUDA Device Count: 1
CUDA Device Name: NVIDIA ...
If CUDA Available shows False, make sure:

You installed the correct CUDA version.

Your GPU drivers are up to date.

The environment is active.

## Features
- Real-time worker detection
- Motion/activity classification (Working/Idle)
- Low-cost hardware compatible
- Auto-start on system boot

## How to Run
```bash
python worker_monitor.py

# Worker Activity Monitoring System

## Overview
This is a vision-based AI system that detects if a worker is actively engaged in their task. It runs standalone and logs activity.
✅ 2. Environment Setup
Goal: All dependency install.

🔹Steps:
Install Python 3.8+

Create a virtual environment (optional but preferred)

python -m venv worker_env
source worker_env/bin/activate  # Windows: worker_env\Scripts\activate
🔹Install Required Libraries

pip install opencv-python numpy torch torchvision onnx onnxruntime
(If optimizing: also install tflite-runtime or tensorrt later)



## Features
- Real-time worker detection
- Motion/activity classification (Working/Idle)
- Low-cost hardware compatible
- Auto-start on system boot

## How to Run
```bash
python worker_monitor.py

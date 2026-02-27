# Smart UAV Scout: End-to-End Object Detection System

## Overview
Smart UAV Scout is a complete Machine Learning pipeline designed for autonomous unmanned aerial vehicles (UAVs). It covers the entire lifecycle of an AI model: from processing aerial datasets (VisDrone) and training a YOLOv8 network in Python, to deploying a highly optimized, real-time C++ inference engine at the edge.

The system is optimized for detecting small objects (vehicles, pedestrians) from top-down aerial perspectives.

## Project Architecture
The repository is divided into two main environments:

1. **Python Training Pipeline:** Data preparation, YOLOv8 training, and ONNX export.
2. **C++ Edge Inference:** High-performance deployment using OpenCV DNN and CMake.

SMART-UAV-SCOUT/
├── cpp_inference/               # C++ Edge Inference Engine
│   ├── src/                     
│   │   └── main.cpp             # Core C++ inference logic
│   ├── CMakeLists.txt           # Build system configuration
│   └── video.mp4                # Test UAV video (optional for repo)
├── models/                      # Lightweight base models (if pushed)
├── scripts/                     # Data preparation and utility scripts
│   └── convert_visdrone.py      # VisDrone to YOLO format converter
├── .gitignore                   # Git exclusion rules
├── my_drone_data.yaml           # Dataset configuration file for YOLO
└── requirements.txt             # Python project dependencies
# Smart UAV Scout: End-to-End Object Detection & Tracking System

## Overview
Smart UAV Scout is a complete Machine Learning pipeline designed for autonomous unmanned aerial vehicles (UAVs). It covers the entire lifecycle of an AI model: from processing aerial datasets (VisDrone) and training a YOLOv8 network in Python, to deploying a highly optimized, real-time C++ inference engine at the edge.

The system is highly optimized for detecting and **tracking** small objects (vehicles, pedestrians) from top-down aerial perspectives in real-time.

## Key Features
* **Custom Aerial Model:** YOLOv8 trained specifically on the VisDrone dataset.
* **Hardware Acceleration:** Zero-copy C++ inference powered by ONNX Runtime, NVIDIA CUDA 13, and cuDNN.
* **Robust Object Tracking:** Custom `SmartTracker` combining IoU (Intersection over Union) and Centroid tracking to maintain stable target IDs, even during object occlusion or intersections.
* **Portable Deployment:** Automated CMake DevOps pipeline that dynamically gathers and links required runtime DLLs for seamless edge deployment.

## Project Architecture
The repository is divided into two main environments:

1. **Python Training Pipeline:** Data preparation, YOLOv8 training, and ONNX export.
2. **C++ Edge Inference:** High-performance deployment using ONNX Runtime, OpenCV, and automated CMake.

SMART-UAV-SCOUT/
├── cpp_inference/               # C++ Edge Inference Engine
│   ├── src/                     
│   │   └── main.cpp             # Core inference, Letterboxing, NMS, and Smart Tracking
│   ├── third_party/             # Local dependencies for automated builds
│   │   ├── onnxruntime/         # ONNX Runtime (CUDA 13 optimized)
│   │   └── cudnn/               # cuDNN libraries for local deployment
│   ├── CMakeLists.txt           # Build system with auto-copy DevOps scripts
│   └── video.mp4                # Test UAV video (optional for repo)
├── models/                      # Lightweight base ONNX models
├── scripts/                     # Data preparation and utility scripts
│   └── convert_visdrone.py      # VisDrone to YOLO format converter
├── .gitignore                   # Git exclusion rules
├── my_drone_data.yaml           # Dataset configuration file for YOLO
└── requirements.txt             # Python project dependencies
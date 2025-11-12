# 🚶‍♂️ MOTS_Mask-RCNN
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-orange)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)]()

---

## 📖 Project Description

This project implements a **robust person tracking system** by combining **Faster R-CNN** for object detection and a **Siamese network** for person re-identification (ReID). The system continuously tracks a pedestrian across frames in a video sequence, even under challenging conditions such as **occlusions**, **lighting changes**, and **similar appearances** among individuals.
The tracking pipeline achieves both accuracy and efficiency by leveraging **deep learning models**, **data augmentation**, and **GPU-accelerated training**.

---

## ✨ Key Features

- **Faster R-CNN with ResNet50 Backbone**  
  Fine-tuned for pedestrian detection using the **MOT16-02** dataset, achieving a training loss of ~1.0065 after 10 epochs.

- **Siamese Network for Person Re-Identification**  
  Learns feature embeddings to uniquely identify individuals across frames using **triplet loss** and fine-tuning on **Market1501**.

- **Bounding Box Tracking with Motion Prediction**  
  Predicts future locations based on prior bounding box velocities, maintaining continuity even under brief occlusions.

- **Similarity-Based Data Association**  
  Combines IoU and embedding similarity scores for robust identity matching between frames.

- **GPU-Accelerated Training**  
  Fully utilizes CUDA with mixed precision and gradient accumulation for optimized performance.

- **Augmented Datasets for Robustness**  
  Includes color jitter, Gaussian blur, and brightness variation to enhance generalization under varied lighting and visual conditions.

---

## 🧠 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Programming Language** | Python 3.10+ |
| **Deep Learning Framework** | PyTorch, Torchvision |
| **Computer Vision** | OpenCV, PIL |
| **Data Handling & Visualization** | Pandas, NumPy, scikit-image, Matplotlib |
| **Datasets** | MOT16, Market1501 |
| **Hardware** | CUDA-enabled GPU |

---

## 🎥 Project Demo

[Watch a full demonstration of the project here.](YOUR_VIDEO_LINK_HERE)

---

## 🏗️ Project Architecture

📂 **Project Root**
```
├── Faster_RCNN_GPU.py              # Training script for Faster R-CNN on MOT16
├── Siamese_network.py              # Core Siamese model (Triplet-based ReID)
├── siamese_network_final_v2_prerit.py  # Advanced Siamese training with augmentation
├── inference_test.py               # Inference and tracking pipeline
├── Project Report.pdf              # System design, methodology, and results
└── outputs/
    ├── faster_rcnn_mots16.pth      # Trained detection model
    ├── finetuned_siamese_model_test.pth  # Trained re-ID model
    └── tracked_output_test.mp4     # Output tracking video
```

### 🧩 Pipeline Overview

1. **Detection (Faster R-CNN)** — Detects pedestrians in each frame.  
2. **Feature Embedding (Siamese Network)** — Extracts a 256-dimensional feature vector per detection.  
3. **Similarity Calculation** — Compares embeddings between frames to match identities.  
4. **Bounding Box Prediction (Tracker)** — Predicts next-frame positions using velocity estimation.  
5. **Data Association** — Combines IoU and embedding similarity to maintain consistent IDs.

---

## ⚙️ Getting Started

### 🧰 Prerequisites

Ensure you have the following installed:
- Python ≥ 3.10  
- CUDA-compatible GPU  
- PyTorch ≥ 2.0  
- Torchvision ≥ 0.15  
- OpenCV ≥ 4.5  

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mots-person-tracking.git
cd mots-person-tracking

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision opencv-python pandas scikit-image tqdm matplotlib
```

---

## 🔐 Configuration

Create a `.env` file in the project root to specify dataset and output paths:

```bash
# Dataset and model paths
MOT16_TRAIN_DIR=D:\Path\To\MOT16\train\MOT16-02
MARKET1501_DIR=D:\Path\To\Market1501
OUTPUT_VIDEO=tracked_output_test.mp4
```

---

## 🚀 Usage

### 1. Train the Detection Model
```bash
python Faster_RCNN_GPU.py
```
This trains a **Faster R-CNN (ResNet50-FPN)** on the **MOT16-02** dataset and saves the model as `faster_rcnn_mots16.pth`.

### 2. Train the Siamese Re-Identification Network
```bash
python siamese_network_final_v2_prerit.py
```
This trains and fine-tunes the **Siamese network** using **triplet loss** across multiple MOT16 sequences and **Market1501**.

### 3. Run Inference and Tracking
```bash
python inference_test.py
```
The script will:
- Detect people per frame  
- Match them using re-ID embeddings  
- Track them across frames  
- Export an annotated output video (`tracked_output_test.mp4`)

---

## 📊 Results Summary

| Model | Dataset | Epochs | Train Loss | Valid Loss | Notes |
|--------|----------|---------|-------------|-------------|--------|
| Faster R-CNN | MOT16-02 | 10 | 1.0065 | — | Fine-tuned from pretrained ResNet50 |
| Siamese Network | Market1501 + MOT16 | 10 | 0.1401 | 0.2154 | Fine-tuned with augmentation |

---

## 🔮 Future Work

- Train both models on the **complete MOT16 dataset** for broader scene understanding.  
- Optimize similarity thresholds for dynamic environments.  
- Integrate **Kalman filters** or **DeepSORT** for enhanced temporal consistency.  
- Explore **multi-camera tracking** and **cross-view re-identification**.

---


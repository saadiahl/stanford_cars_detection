# Stanford Cars Detection 🚗🔍

![YOLO](https://img.shields.io/badge/YOLOv5-Object%20Detection-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Stanford Cars](https://img.shields.io/badge/Dataset-Stanford%20Cars-orange)

## 📌 Overview

This repository provides a complete pipeline for converting the **Stanford Cars Dataset** into the **YOLO format**, performing **stratified splitting**, and training an **object detection model** using **YOLOv5**.  

🚀 **Key Features:**
- Converts **Stanford Cars Dataset** annotations to YOLO format.
- Splits **cars_test** into **50% validation, 25% test, 25% validation** using stratified sampling.
- Organizes images and labels according to the **Ultralytics YOLO dataset structure**.
- Generates a **stanford_cars.yaml** configuration file for training YOLO.
- Supports **training YOLOv5 on Mac M2 (MPS support)**.

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/saadiahl/stanford_cars_detection.git
cd stanford_cars_detection
```

### 2️⃣ Install dependencies
``` bash
pip install -r requirements.txt
```

### 3️⃣ Download and extract the Stanford Cars Dataset
``` python
import kagglehub
# Download latest version
path = kagglehub.dataset_download("rickyyyyyyy/torchvision-stanford-cars")

print("Path to dataset files:", path)
```

## 🛠️ Usage

### 🔹 Convert dataset to YOLO format
```
python data_preprocess.py --dataset_root /path/to/stanford_cars
```
### 🔹 Train YOLO model 
```
yolo detect train data=/path/to/stanford_cars.yaml model=best.pt epochs=100 imgsz=640 batch=24 patience=10 degrees=15 single_cls=True device=mps
Note: The --device mps flag enables training on Mac M2.
```
## 📊 Results

| Model  | mAP@50 |  mAP75 | mAP@50-95|
|--------|--------|----------|----------------|
| -------- | 0.9373129779298989 | 0.9924700662373929    | 0.9891402017131272 |
| IoU threshold | 0.5    | 0.75      | 0.5 - 0.95            |


mAP50: 0.9373129779298989 (mean Average Precision at IoU threshold 0.5)
mAP75: 0.9924700662373929(mean Average Precision at IoU threshold 0.75)
mAP50-95: 0.9891402017131272(mean Average Precision across multiple IoU thresholds from 0.5 to 0.95)

## 🔗 References
[Stanford Cars Dataset](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars)

[Ultralytics YOLO Documentation](https://docs.ultralytics.com)


This `README.md` includes:
- **Project Overview**
- **Dataset Structure**
- **Installation Steps**
- **Usage Instructions**
- **Training Instructions**
- **Results Table (TBD)**
- **References**




# 🌿 Moringa Disease Classification using Deep Learning

![Project Banner](https://img.shields.io/badge/DeepLearning-ComputerVision-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-blue)

---

## 📌 Project Overview

This project focuses on the **automated classification of Moringa leaf diseases** using state-of-the-art **Deep Learning models**.
It aims to assist farmers, researchers, and agricultural experts in **early disease detection** to improve crop health and productivity.

---

## 🎯 Objectives

* Develop a robust image classification system for Moringa leaf diseases
* Compare multiple deep learning architectures
* Improve classification accuracy using transfer learning
* Provide a scalable solution for real-world agricultural use

---

## 🧠 Models Used

This project implements and compares the following architectures:

* 📱 MobileNetV3 (Lightweight & Efficient)
* ⚡ EfficientNet-B0 (Optimized Performance)
* 🧱 ResNet50 (Deep Residual Learning)
* 🔍 Vision Transformer (ViT / Swin-T)

---

## 📂 Project Structure

```
moringa-disease-classification-dl/
│
├── dataset/
│   ├── healthy/
│   ├── disease_1/
│   ├── disease_2/
│
├── models/              # Saved trained models
├── results/             # Graphs, metrics, outputs
├── notebooks/           # Google Colab notebooks
├── src/                 # Source code (training, utils)
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

* Images collected manually and/or from open datasets
* Preprocessed and labeled into categories:

  * Healthy
  * Diseased (multiple classes)

### 🖼️ Sample Data

(Add sample images here later)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/moringa-disease-classification-dl.git
cd moringa-disease-classification-dl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Run Training

```bash
python src/train.py
```

### ▶️ Run Prediction

```bash
python src/predict.py --image path_to_image
```

---

## 📈 Results

* Model performance evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1-score

📊 Training graphs and results are stored in the `/results` folder.

---

## 💾 Saving Outputs

The project saves:

* ✅ Trained models (`.h5`, `.pt`)
* ✅ Training history (`.json`)
* ✅ Performance plots (`.png`)

---

## 🌍 Future Work

* Deploy as a web/mobile application
* Real-time disease detection using camera
* Expand dataset for better generalization
* Integrate with IoT-based smart farming systems

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Surpasstech**

* AI & Deep Learning Enthusiast
* Focused on Agricultural AI Solutions

---

## ⭐ Support

If you find this project useful, please ⭐ star the repository!

---

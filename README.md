# 🚦 Monitoring Traffic Congestion in Smart Cities Using CNN

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Made with Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![GitHub Repo Stars](https://img.shields.io/github/stars/Sairam-kattunga/Traffic_Congestion_Monitoring_CNN?style=social)](https://github.com/Sairam-kattunga/Traffic_Congestion_Monitoring_CNN/stargazers)

</div>

---

## 📌 Overview

This project provides a **deep learning-based solution** for **real-time traffic congestion monitoring** in smart cities.  
By leveraging **Convolutional Neural Networks (CNNs)** and **transfer learning**, the system classifies road images into:  

- 🟥 **High Congestion**  
- 🟩 **Low Congestion**  

We benchmark a **custom CNN** against **VGG16, ResNet50, and MobileNetV2** to identify the most accurate model.

---

## 🌟 Key Features

- 📷 **Traffic Congestion Detection** → High/Low congestion classification from static images.  
- 🔍 **Model Benchmarking** → Compare custom CNN vs transfer learning models.  
- ♻ **Transfer Learning** → Pre-trained networks for improved accuracy.  
- 🖼 **Data Augmentation** → Prevents overfitting and enhances robustness.  
- 🧩 **Modular Codebase** → Easy-to-read, reproducible training and evaluation scripts.  
- 🌐 **Web App** → Simple Flask/Streamlit-based demo to test images in real-time.  

---

## 📊 Results Sneak-Peek

**🏆 Best Model:** `ResNet50 (Transfer Learning)` → **~99% Accuracy**

![ResNet50 Confusion Matrix](results/resnet50_transfer_confusion_matrix.png)

| Model                   | Accuracy | Notes                   |
| ----------------------- | :------: | ----------------------- |
| Custom CNN              |   89%    | Baseline model          |
| VGG16 (Transfer)        |   96%    | Significant improvement |
| MobileNetV2 (Transfer)  |   98%    | High efficiency         |
| **ResNet50 (Transfer)** | **99%**  | Best performer          |

---

## 🛠 Project Workflow

1. **Data Preprocessing** → Clean, split & augment dataset.  
2. **Model Training** → Train custom CNN & transfer learning models.  
3. **Evaluation** → Generate accuracy scores, loss plots & confusion matrices.  
4. **Deployment** → Simple web app (Flask/Streamlit) for demo usage.  

---

## 📂 Repository Structure

```

traffic\_congestion\_project/
├── dataset/                # Image dataset
│   ├── High\_Congestion/
│   └── Low\_Congestion/
├── models/                 # Saved .keras models
├── results/                # Confusion matrices, plots
├── scripts/                # Training & evaluation scripts
│   ├── preprocess\_data.py
│   ├── train\_cnn.py
│   ├── train\_transfer.py
│   ├── evaluate.py
│   └── app.py              # Web application (Flask/Streamlit)
├── README.md
└── requirements.txt

```

---

## ⚙ Getting Started

### 1️⃣ Dataset Setup
- Download dataset from Kaggle:  
  👉 [Traffic Management - Image Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)  
- Organize folders as:
```

dataset/
├── High\_Congestion/
└── Low\_Congestion/

````
- Place **dense traffic** images in `High_Congestion` and **sparse traffic** in `Low_Congestion`.

---

### 2️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Sairam-kattunga/Traffic_Congestion_Monitoring_CNN.git
cd Traffic_Congestion_Monitoring_CNN

# Install dependencies
pip install -r requirements.txt
````

---

### 3️⃣ Train Models

```bash
cd scripts

# Train custom CNN
python train_cnn.py

# Train transfer learning models
python train_transfer.py --model vgg16
python train_transfer.py --model resnet50
python train_transfer.py --model mobilenet
```

---

### 4️⃣ Evaluate Models

```bash
python evaluate.py
```

This generates:

* ✅ Accuracy & loss plots
* ✅ Confusion matrices
* ✅ Model performance comparison

---

### 5️⃣ Run Web App 🚀

We included a simple **Flask/Streamlit app** (`app.py`) for testing on custom images.

```bash
# Inside scripts folder
python app.py
```

* 🌐 Open browser → `http://127.0.0.1:5000/` (Flask)
* Or → Streamlit will auto-launch at `http://localhost:8501/`

Upload an image → Get **Congestion Prediction** instantly ✅

---
### 🖼 Web App Demo Screenshot

Here’s how the prediction page looks in action 👇
<img width="3200" height="2000" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/3591c33f-cd8a-4e14-8e98-de188857f6f6" />



## 💡 Future Enhancements

* 🎥 **Real-time Video Feed** integration with OpenCV.
* ☁ **Cloud Deployment** (AWS / GCP / Heroku).
* 🚗 **Multi-Class Detection** (accidents, roadblocks, fires, etc.).
* 🎯 **Hyperparameter Optimization** with Optuna/KerasTuner.
* 📊 **Dashboard Analytics** for smart city planners.

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 📧 Contact

**Rama Venkata Manikanta Sairam Kattunga**
🌐 [Portfolio](https://simple-portfolio-sigma-orpin.vercel.app/)
📩 [sairamkattunga333@gmail.com](mailto:sairamkattunga333@gmail.com)
📂 [GitHub Repo](https://github.com/Sairam-kattunga/Traffic_Congestion_Monitoring_CNN)

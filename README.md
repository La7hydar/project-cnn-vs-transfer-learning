# 🐾 Cats vs Dogs Classification: CNN from Scratch vs Transfer Learning (MobileNetV2)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Red?logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

Proyek ini mengeksplorasi, mengimplementasikan, dan membandingkan dua pendekatan utama dalam klasifikasi citra berbasis *Deep Learning*: pembangunan arsitektur **Convolutional Neural Network (CNN) kustom dari awal (*from scratch*)** dan pemanfaatan **Transfer Learning (MobileNetV2)** dengan strategi *Feature Extraction*. 

Dataset biner (*Cats vs Dogs*) diekstraksi secara spesifik dari dataset publik **CIFAR-10**.

---

## 📌 Alur Proyek & Preprocessing

1. **Filtering Dataset:** Menyaring secara otomatis kelas Kucing (`label 3`) dan Anjing (`label 5`) dari dataset CIFAR-10 sehingga menghasilkan total 12.000 gambar biner.
2. **Stratified Data Splitting:** Membagi dataset secara proporsional dengan rasio **70% Training (8.400 gambar), 15% Validation (1.800 gambar), dan 15% Testing (1.800 gambar)**.
3. **Normalisasi Piksel:** Mengubah nilai mentah matriks gambar $[0, 255]$ menjadi rentang $[0, 1]$ dengan membaginya dengan 255.0 untuk menstabilkan gradien saat *training*.
4. **Image Resizing:** Khusus untuk arsitektur MobileNetV2, dimensi gambar CIFAR-10 otomatis diperbesar (*resized*) dari $32 \times 32$ piksel menjadi $128 \times 128$ piksel agar kompatibel dengan *input shape* model *pretrained*.

---

## 🏗️ Arsitektur Model

### 1. CNN from Scratch
* **Lapisan Konvolusi:** 2 Layer *Conv2D* bertingkat (32 dan 64 filter) dengan fungsi aktivasi ReLU.
* **Reduksi Dimensi:** Lapisan *MaxPooling2D* ($2 \times 2$) di setiap akhir blok konvolusi.
* **Regularisasi:** Lapisan *Dropout* (0.25) untuk mencegah terjadinya *overfitting*.
* **Lapisan Klasifikasi:** Lapisan *Flatten*, 1 *Dense Layer* tersembunyi (64 neuron, ReLU), dan 1 neuron output dengan fungsi aktivasi *Sigmoid*.

### 2. Transfer Learning (MobileNetV2)
* **Base Model:** Arsitektur MobileNetV2 dengan bobot bawaan prapelatihan dari *ImageNet* (`weights='imagenet'`).
* **Strategi:** *Feature Extraction* di mana seluruh lapisan dasar dibekukan secara total (`trainable = False`).
* **Lapisan Klasifikasi:** Lapisan *GlobalAveragePooling2D*, 1 *Dense Layer* tersembunyi (64 neuron, ReLU), dan 1 neuron output dengan fungsi aktivasi *Sigmoid*.

---

## 📥 Sumber & Tautan Unduhan Dataset

Dataset diakses secara otomatis melalui API internal TensorFlow Keras (`tf.keras.datasets.cifar10`). Namun, jika Anda ingin mengunduh file mentah (*raw dataset*) CIFAR-10 secara langsung, silakan gunakan tautan resmi berikut:

* 🌐 **Halaman Resmi:** [CIFAR-10 Dataset Webpage](https://www.cs.toronto.edu/~kriz/cifar.html)
* 📂 **Download Langsung (Format Python):** [Download Raw CIFAR-10 Dataset (.tar.gz)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

---

## 🛠️ Tech Stack & Library

* **Bahasa Pemrograman:** Python
* **Framework Deep Learning:** TensorFlow / Keras
* **Visualisasi & Analisis Data:** Matplotlib, Seaborn, NumPy, Pandas
* **Metrik Evaluasi:** Scikit-Learn (`confusion_matrix`, `classification_report`)

---

## 📁 Struktur Repositori

```text
├── .gitignore
├── README.md
├── cats_vs_dogs_classification.ipynb   # File Notebook utama (Kaggle/Colab)
└── assets/                             # Folder penyimpanan hasil visualisasi
    ├── cnn_accuracy_loss.png
    ├── cnn_confusion_matrix.png
    ├── tl_accuracy_loss.png
    └── tl_confusion_matrix.png

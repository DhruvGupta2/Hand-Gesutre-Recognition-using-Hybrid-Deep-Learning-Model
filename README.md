# 🖐️ Hand Gesture Recognition with Hybrid Deep Learning

A hybrid deep learning approach combining **Swin Transformer**, **ResNet34**, and **BiLSTM** to recognize **18 static hand gestures** using the [HaGRID](https://github.com/hukenovs/hagrid) dataset with 98.03% accuracy. Built with PyTorch and Hugging Face Transformers.
This project's research paper has also gone for journals.

**Classification report:**
<img src="https://github.com/user-attachments/assets/5adef14a-b2d5-456c-a561-eb93d56db15d" width="600"/>

**Confusion Matrix:**
<img src="https://github.com/user-attachments/assets/d5b28062-341a-4683-9ad8-8f4c4a204c46" width="600"/>

**Train Accuracy per epoch:**
<img src="https://github.com/user-attachments/assets/de82a718-9066-4dc5-98c7-683ab698b352" width="600"/>

**Loss:**
<img src="https://github.com/user-attachments/assets/65a3768f-21f9-460d-a494-27a3d432b657" width="600"/>

---

## 🚀 Overview

This project leverages the strengths of transformer-based and convolutional neural networks for highly accurate hand gesture classification.

**Hybrid Architecture:**
- 🧠 **Swin Transformer** – Token-based visual representation.
- 🧠 **ResNet34** – Hierarchical CNN-based features.
- 🔁 **BiLSTM** – Sequential modeling of combined features.

**Website:**
<img src="https://github.com/user-attachments/assets/e53cb191-d400-45e9-af81-134978e0c007" width="600"/>

---

## 🗂️ Dataset

- **Name:** [HaGRID (Hand Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid)
- **Size:** 120K+ gesture images
- **Classes:** 18 gesture categories
- **Input Format:** `.jpg` images with per-class annotation in `.json` files.

---

**Number of images per classes:**
<img src="https://github.com/user-attachments/assets/e53cb191-d400-45e9-af81-134978e0c007" width="600"/>

## 🏗️ Model Architecture

**Architecture:**
<img src="https://github.com/user-attachments/assets/177f8a91-2cee-4bba-a6ff-e81df31d7459" width="800"/>

```txt
Input Image (224x224)
│
├──> Swin Transformer (Feature Vector: 768)
├──> ResNet34         (Feature Vector: 512)
│
└──> Concatenation → BiLSTM (bidirectional)
     ↓
     Fully Connected Layer → Softmax (18 classes)



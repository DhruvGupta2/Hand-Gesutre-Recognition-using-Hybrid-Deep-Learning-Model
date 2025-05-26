# 🖐️ Hand Gesture Recognition with Hybrid Deep Learning

A hybrid deep learning approach combining **Swin Transformer**, **ResNet34**, and **BiLSTM** to recognize **18 static hand gestures** using the [HaGRID](https://github.com/hukenovs/hagrid) dataset with 98.03% accuracy. Built with PyTorch and Hugging Face Transformers.

---

## 🚀 Overview

This project leverages the strengths of transformer-based and convolutional neural networks for highly accurate hand gesture classification.

**Hybrid Architecture:**
- 🧠 **Swin Transformer** – Token-based visual representation.
- 🧠 **ResNet34** – Hierarchical CNN-based features.
- 🔁 **BiLSTM** – Sequential modeling of combined features.

---

## 🗂️ Dataset

- **Name:** [HaGRID (Hand Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid)
- **Size:** 120K+ gesture images
- **Classes:** 18 gesture categories
- **Input Format:** `.jpg` images with per-class annotation in `.json` files.

---

## 🏗️ Model Architecture

```txt
Input Image (224x224)
│
├──> Swin Transformer (Feature Vector: 768)
├──> ResNet34         (Feature Vector: 512)
│
└──> Concatenation → BiLSTM (bidirectional)
     ↓
     Fully Connected Layer → Softmax (18 classes)
![WhatsApp Image 2025-05-26 at 07 56 48_0ada890d](https://github.com/user-attachments/assets/8f1dc323-a116-4716-af53-ffd2ed86053d)


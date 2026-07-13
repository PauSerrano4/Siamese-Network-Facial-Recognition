# Siamese Network for Facial Recognition 🧠👤

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

A facial verification system built with a **Siamese Neural Network** on top of a pre-trained **ResNet18** backbone, trained with **Contrastive Loss** to learn whether two face images belong to the same person.

## 🎯 Project Goal

Instead of training a classifier limited to a fixed set of identities, this project learns a general **similarity metric** between face images: given two photos, the network outputs how close they are in embedding space, so it can generalize to identities it has never seen during training.

## 🔍 Exploratory Data Analysis

Before training, the dataset (31 celebrities, ~30-120 images each) was analyzed to catch issues that could bias the model:
- **Class imbalance**: image counts per identity vary significantly, addressed with data augmentation on minority classes.
- **Image resolution & aspect ratio**: scanned across the full dataset to decide on a consistent resizing/cropping strategy.
- **Brightness distribution**: measured average pixel intensity per image to detect lighting bias (e.g., an identity photographed mostly in dark scenes), which justified using `ColorJitter` during augmentation.

## 🏗️ Architecture & Training

| Component | Choice |
|---|---|
| Backbone | ResNet18 (ImageNet pre-trained, transfer learning) |
| Output head | Removed final FC layer → raw embedding vectors |
| Loss function | Contrastive Loss (margin-based) |
| Optimizer | Adam |
| Data split | 70% train / 15% validation / 15% test (stratified) |
| Augmentation | Random flips, rotations, ColorJitter |

The model processes pairs of images through **shared weights** (`forward_once`), and the Contrastive Loss pulls embeddings of the same identity closer together while pushing different identities apart, based on their Euclidean distance.

## 📂 Repository Structure

```
├── project3.ipynb        # Full pipeline: EDA → preprocessing → training → evaluation
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/PauSerrano4/siamese-facial-recognition.git
cd siamese-facial-recognition
pip install -r requirements.txt
jupyter notebook project3.ipynb
```

> Note: the dataset (`Dataset.csv` + `Faces/` image folder) is not included in this repo due to size. Download it from Kaggle: [Face Recognition Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset) (31 classes), and place it in the repo root before running the notebook.

## 🔮 Future Improvements

- Add automatic **face detection & cropping** (MTCNN or Haar Cascades) before feeding images to the network, to remove background noise and improve accuracy.
- Experiment with **Triplet Loss** instead of Contrastive Loss for potentially better embedding separation.
- Fine-tune deeper ResNet layers instead of only the head, once the dataset is large enough to avoid overfitting.

## 🛠️ Tech Stack

Python · PyTorch · torchvision · OpenCV · pandas · scikit-learn · seaborn/matplotlib

## ✍️ Author

**Pau Serrano** — Computer Engineering student @ UPC-FIB
[GitHub](https://github.com/PauSerrano4) · [LinkedIn](https://www.linkedin.com/in/pau-serrano-b35215329)

# ML-Animal-Classification-Model  
**Lightweight CNN for 5-Class Animal Recognition**

---

## 📊 Overview

- Built and trained a compact **Convolutional Neural Network (CNN)** to classify images of **cats, dogs, horses, elephants, and lions**.
- Achieves **≈ 98 % validation accuracy** on a balanced dataset with only ~150 K trainable parameters, making it suitable for edge devices.
- Implements a fast preprocessing pipeline using **Keras `ImageDataGenerator`** with on-the-fly normalization and augmentation.
- Saves the trained model (`.h5`), training history (`.npy`), and class-index mapping for easy deployment or transfer learning.

---

## 🗂️ Dataset Structure

```
animals/
├── train/
│   ├── cat/
│   ├── dog/
│   ├── horse/
│   ├── elephant/
│   └── lion/
├── val/
│   ├── cat/
│   ├── dog/
│   ├── horse/
│   ├── elephant/
│   └── lion/
└── inf/
    ├── cat/
    ├── dog/
    ├── horse/
    ├── elephant/
    └── lion/
```

Each class contains an equal number of images for training and validation. The `inf` folder is used for inference testing with one image per class.
```

---

### 🛠 Tools Used (Optional but recommended)

````markdown
## 🛠 Tools Used

- **Python**: NumPy, TensorFlow (Keras)
- **Keras `ImageDataGenerator`** for preprocessing
- **Jupyter Notebook / VS Code** for experimentation and visualization
- **Matplotlib** (if training curves are plotted)



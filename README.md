# 👁️ Optic Disc and Optic Cup Segmentation

A deep learning-based **medical image segmentation** project for automatic segmentation of the **optic disc (OD)** and **optic cup (OC)** from retinal fundus images.

The project explores multiple segmentation architectures, including a **custom PyTorch U-Net**, **U-Net**, **U-Net++**, and **SegFormer**, with EfficientNet-B7 used as the encoder for the segmentation-models-pytorch implementations.

---

## 🎯 Project Overview

Accurate segmentation of the optic disc and optic cup is an important step in retinal image analysis and glaucoma screening.

The **cup-to-disc ratio (CDR)** is a clinically relevant measurement derived from the optic cup and optic disc. Automating their segmentation can support consistent and scalable retinal image analysis.

This project investigates how different segmentation architectures, preprocessing techniques, and optimization strategies affect optic disc and optic cup segmentation performance.

### Main objectives

- Segment the optic disc (OD) and optic cup (OC)
- Compare different segmentation architectures
- Evaluate the impact of image preprocessing
- Compare different optimizers
- Analyse segmentation performance using IoU and Dice
- Visualize predicted masks against ground-truth masks

---

## 📚 Datasets

The project uses three publicly available retinal fundus image datasets:

- **G1020**
- **ORIGA**
- **REFUGE**

The notebooks load images and corresponding segmentation masks from:

```text
G1020/
ORIGA/
REFUGE/
```

Images and masks are resized to **256 × 256 pixels** and the segmentation masks are converted to a three-class representation.

The project uses an **80% training, 10% validation, and 10% test split**.

---

## 🔧 Preprocessing

The preprocessing pipeline includes several image enhancement and augmentation techniques.

### Image Enhancement

**Bilateral Filtering**

Used to reduce image noise while preserving important edges and structural details.

**CLAHE**

Contrast Limited Adaptive Histogram Equalization is applied to improve local contrast, particularly around retinal structures with subtle boundaries.

### Augmentation

The pipeline also applies:

- Random horizontal/vertical flipping
- Random rotation up to ±15°
- Image normalization
- Resizing to 256 × 256

The corresponding masks are transformed together with the images to maintain spatial alignment.

---

## 🧠 Segmentation Pipeline

```text
                  Retinal Fundus Image
                           ↓
                  Bilateral Filtering
                           ↓
                         CLAHE
                           ↓
                   Resize to 256×256
                           ↓
                Flip / Rotation / Normalize
                           ↓
                    Segmentation Model
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        Custom U-Net      U-Net        U-Net++
             │        EfficientNet-B7  EfficientNet-B7
             │             │             │
             └─────────────┼─────────────┘
                           ↓
                       SegFormer
                    EfficientNet-B7
                           ↓
                  Pixel-wise Prediction
                           ↓
                ┌──────────┴──────────┐
                ↓                     ↓
          Optic Disc (OD)       Optic Cup (OC)
                ↓                     ↓
                └──────────┬──────────┘
                           ↓
                  Segmentation Masks
```

---

## 🤖 Models

### 1. Custom PyTorch U-Net

A U-Net architecture implemented directly in PyTorch.

The custom architecture contains:

- Encoder blocks
- Bottleneck
- Decoder blocks
- Skip connections
- Batch normalization
- ReLU activations
- Dropout
- Final convolutional segmentation layer

The implementation was built from scratch rather than using a pretrained segmentation architecture.

---

### 2. U-Net + EfficientNet-B7

A U-Net model implemented using `segmentation_models_pytorch` with:

- EfficientNet-B7 encoder
- ImageNet pretrained encoder weights
- 3-channel input
- 3 segmentation classes

---

### 3. U-Net++ + EfficientNet-B7

A U-Net++ implementation using:

- EfficientNet-B7 encoder
- ImageNet pretrained encoder weights
- Nested skip connections
- 3-class segmentation output

---

### 4. SegFormer + EfficientNet-B7

A SegFormer-based segmentation model using:

- EfficientNet-B7 encoder
- ImageNet pretrained encoder weights
- 3-channel input
- 3-class segmentation output

---

## ⚙️ Training Configuration

The segmentation-models-pytorch experiments use:

| Parameter | Configuration |
|---|---|
| Input Size | 256 × 256 |
| Input Channels | 3 |
| Classes | 3 |
| Encoder | EfficientNet-B7 |
| Encoder Weights | ImageNet |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 50 |
| LR Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 3 |
| Minimum Learning Rate | 1e-5 |

The custom PyTorch U-Net notebook uses a smaller batch size and training configuration:

| Parameter | Configuration |
|---|---|
| Batch Size | 8 |
| Epochs | 25 |
| Initial Learning Rate | 0.001 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau |

---

## 📉 Loss Functions

The training objective combines two segmentation losses:

```text
Total Loss
    │
    ├── Dice Loss
    │
    └── Jaccard / IoU Loss
```

The combined loss is designed to optimize overlap between predicted and ground-truth segmentation masks.

---

## 📊 Evaluation Metrics

The models are evaluated using:

- **Intersection over Union (IoU)**
- **Dice Coefficient / F1 Score**
- Precision
- Recall
- Specificity
- Accuracy

The notebooks calculate these metrics using `segmentation_models_pytorch`.

---

## 🏆 Results

The project report compares the segmentation architectures using the Adam optimizer with preprocessing.

| Model | IoU | Dice | Accuracy |
|---|---:|---:|---:|
| Custom U-Net | 0.9933 | 0.9966 | 0.9977 |
| U-Net + EfficientNet-B7 | 0.9942 | 0.9971 | 0.9981 |
| UNet++ + EfficientNet-B7 | 0.9910 | 0.9955 | 0.9970 |
| **SegFormer + EfficientNet-B7** | **0.9943** | **0.9971** | **0.9981** |

**Key result:** SegFormer achieved the highest reported IoU of **0.9943**, while both SegFormer and U-Net with EfficientNet-B7 achieved the highest reported accuracy of **0.9981**. :contentReference[oaicite:2]{index=2}

The project also found that Adam provided stable convergence compared with SGD and RMSProp in the optimizer experiments. :contentReference[oaicite:3]{index=3}

---

## 🔬 Optimizer Comparison

The custom U-Net was evaluated using:

- Adam
- SGD
- RMSProp

The reported results were:

| Configuration | IoU | Dice | Accuracy |
|---|---:|---:|---:|
| Adam + Preprocessing | 0.9933 | 0.9966 | 0.9977 |
| Adam + No Preprocessing | 0.9936 | 0.9968 | 0.9979 |
| SGD + Preprocessing | 0.9757 | 0.9877 | 0.9917 |
| SGD + No Preprocessing | 0.9761 | 0.9879 | 0.9919 |
| RMSProp + Preprocessing | 0.9932 | 0.9966 | 0.9977 |
| RMSProp + No Preprocessing | 0.9928 | 0.9964 | 0.9976 |

These experiments demonstrate the effect of optimizer selection on convergence and segmentation performance. :contentReference[oaicite:4]{index=4}

---

## 🖼️ Qualitative Results

The notebooks visualize segmentation predictions using:

```text
Original Fundus Image
          ↓
Ground Truth Mask
          ↓
Predicted Mask
          ↓
Prediction Overlay
```

This provides a visual comparison between the model predictions and the expert-annotated masks, particularly around optic disc and optic cup boundaries. :contentReference[oaicite:5]{index=5}

---

## 📂 Project Structure

```text
Optic-Disc-and-Optic-Cup-Segmentation/
│
├── OD_OC_UNET.ipynb
│   └── U-Net with EfficientNet-B7 encoder
│
├── OD_OC_UNET_Plus_Plus.ipynb
│   └── U-Net++ with EfficientNet-B7 encoder
│
├── OD_OC_UNET_SegFormer.ipynb
│   └── SegFormer with EfficientNet-B7 encoder
│
├── OD_OC_UNET_PyTorch_Self.ipynb
│   └── Custom PyTorch U-Net implementation
│
├── PyTorch_UNET_Architecture.jpg
│   └── Custom U-Net computation graph
│
├── pyproject.toml
│   └── Project dependencies and configuration
│
├── uv.lock
│   └── Locked Python dependencies
│
└── README.md
```

---

## 🛠️ Technologies

`Python` `PyTorch` `OpenCV`

`segmentation-models-pytorch`

`U-Net` `U-Net++` `SegFormer`

`EfficientNet-B7`

`NumPy` `Pandas` `Matplotlib`

`torchinfo` `torchviz`

`uv`

---

## 🔁 Reproducibility

The notebooks use a fixed random seed of **42** for Python, NumPy, and PyTorch, with deterministic CUDA/cuDNN settings enabled where applicable.

This helps make training and evaluation more reproducible across runs.

---

## 🚀 Key Learning Areas

This project demonstrates practical experience with:

- Medical image segmentation
- Retinal fundus image analysis
- Semantic segmentation
- CNN-based segmentation
- Transformer-based segmentation
- Custom PyTorch model development
- Transfer learning
- EfficientNet encoders
- Image preprocessing and augmentation
- Segmentation loss functions
- Model evaluation
- Qualitative visualization
- Experiment comparison

---

## 🔮 Future Work

Potential extensions include:

- Lightweight models for real-time deployment
- Class-wise segmentation evaluation
- Testing on more diverse datasets
- Cup-to-disc ratio (CDR) computation
- Glaucoma risk classification
- Hybrid architectures such as TransUNet
- Improved generalization across datasets

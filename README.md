# MT-FINet
## Introduction
MT-FINet (Multi-Task Feature Interaction Network) is a multimodal, multi-task deep learning framework that jointly predicts central lymph node metastasis (CLNM) and lateral lymph node metastasis (LLNM) in papillary thyroid carcinoma (PTC) patients. The model integrates preoperative tumor ultrasound (US) images, C6-level paravertebral adipose tissue (PVAT) CT images, and clinical variables through a shared ResNet50-based encoder.

A two-stage **AutoAlign-EM** feature interaction module, inspired by the Expectation-Maximization algorithm, aligns task-specific representations through a set of learnable latent basis vectors, allowing knowledge to be shared between the CLNM and LLNM tasks while preserving task-specific characteristics. Task-specific heads then output the CLNM and LLNM predictions.

## Requirements
* python 3.11.4
* pytorch 2.1.0+cu121
* torchvision
* pandas
* numpy
* scikit-learn
* Pillow

## Project Structure
```
MT-FINet/
├── models/
│   └── ResNet50MultiTask.py
├── utils/
│   └── MultiTaskDataset.py
├── Results/
│   └── weights
├── train.py
├── predict.py
└── README.md
```

## Usage
### 1. Train MT-FINet
Training images use ResNet50 (first convolutional layer expanded to accept 6-channel input, i.e. concatenated US and CT images) with online data augmentation.



### 2. Predict CLNM / LLNM
To generate predictions on the internal/external test set:

Runs inference on the test set, and reports AUC, accuracy, sensitivity, and specificity (with bootstrap 95% confidence intervals) for both CLNM and LLNM.

Fused image and clinical features from the task-specific heads (`cclnm_feat`, `lclnm_feat`) are also returned for downstream analysis.

## Model Architecture
**Shared encoder:** a 6-channel ResNet50 backbone extracts joint features from concatenated tumor US and PVAT CT images; a lightweight fully connected encoder embeds clinical variables. Image and clinical features are concatenated and fused into a unified representation.

**Two-stage feature interaction (AutoAlign-EM):** the shared representation is projected into task-specific subspaces for CLNM and LLNM, then refined through two stacked EM-inspired alignment stages that iteratively update a set of learnable basis vectors (Expectation step: soft assignment via attention; Maximization step: weighted basis update), with residual connections preserving task-specific information.

**Task-specific heads:** two shallow MLP heads output the binary probabilities for CLNM and LLNM.

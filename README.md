# Custom CNN Architecture for CIFAR-10

## Project Overview
This project implements a custom Convolutional Neural Network (CNN) architecture for the CIFAR-10 dataset, focusing on efficient design with specific architectural constraints and modern training techniques.

## Model Architecture
The network has the following key features:

### Layer Structure
1. **C1 Block (Regular Convolution)**
   - Conv2d(3→32, 3x3) + BN + ReLU
   - Conv2d(32→48, 5x5) + BN + ReLU

2. **C2 Block (Depthwise Separable Convolution)**
   - Depthwise Conv2d(48→48, 3x3)
   - Pointwise Conv2d(48→64, 1x1)
   - BN + ReLU
   - Conv2d(64→64, 3x3) + BN + ReLU

3. **C3 Block (Dilated Convolution)**
   - Dilated Conv2d(64→64, 3x3, dilation=4) + BN + ReLU

4. **C4 Block (Strided Convolution)**
   - Conv2d(64→96, 3x3, stride=2) + BN + ReLU

### Output Layer
- Global Average Pooling
- Fully Connected Layer (96→10)

## Key Features
- **Receptive Field**: 47x47 (exceeds required 44x44)
- **Parameters**: < 200k
- **Special Convolutions**: 
  - Depthwise Separable Convolution
  - Dilated Convolution
  - Strided Convolution (replacing MaxPooling)

## Data Augmentation
Using Albumentations library with:
1. Horizontal Flip (p=0.5)
2. ShiftScaleRotate
3. CoarseDropout
   - max_holes=1
   - max_height=16px
   - max_width=16px
   - min_holes=1
   - min_height=16px
   - min_width=16px
   - fill_value=CIFAR_MEAN

## Training Details
- **Optimizer**: SGD with Nesterov Momentum
  - Learning Rate: 0.015
  - Momentum: 0.9
  - Weight Decay: 1e-4
- **Learning Rate Schedule**: Cosine Annealing with Warm-up
- **Loss Function**: Negative Log Likelihood
- **Target Accuracy**: 85%

![WhatsApp Image 2024-12-28 at 02 04 42_feeb5e7b](https://github.com/user-attachments/assets/66528ab2-2207-4396-8b10-2bd413c08f68)

## Dataset
CIFAR-10:
- 50,000 training images
- 10,000 testing images
- 10 classes
- 32x32 RGB images

# Receptive Field Calculation

Input: 1x1
Conv1 (3x3): 3x3
Conv1 (5x5): 7x7
Conv2 (3x3): 9x9
Conv2 (1x1): 9x9
Conv2 (3x3): 11x11
Conv3 (3x3, dilation=4): 27x27
Conv4 (3x3, stride=2): 47x47

# Training Logs

<img width="542" alt="image" src="https://github.com/user-attachments/assets/e78b9d78-cc9a-4b4c-8065-d760b2831890" />



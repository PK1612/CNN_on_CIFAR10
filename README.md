# Custom CNN Architecture for CIFAR-10

## Project Overview
This project implements a custom Convolutional Neural Network (CNN) architecture for the CIFAR-10 dataset, focusing on efficient design with specific architectural constraints and modern training techniques.

## Model Architecture
The network follows a C1C2C3C4 architecture with the following key features:

### Layer Structure
1. **C1 Block (Regular Convolution)**
   - Conv2d(3→16, 3x3) + BN + ReLU
   - Conv2d(16→32, 5x5) + BN + ReLU

2. **C2 Block (Depthwise Separable Convolution)**
   - Depthwise Conv2d(32→32, 3x3)
   - Pointwise Conv2d(32→48, 1x1)
   - BN + ReLU
   - Conv2d(48→48, 3x3) + BN + ReLU

3. **C3 Block (Dilated Convolution)**
   - Dilated Conv2d(48→64, 3x3, dilation=4) + BN + ReLU

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

# CNN Model Architecture
- The model used in this project is a Convolutional Neural Network (CNN) model. 
- The model consists of 4 convolutional blocks, followed by Global Average Pooling and finally fully collected layer.
- The model consists following type of layers in different convolutional blocks:
    - Normal Conv2D layers
    - Conv2D layers with dilation
    - Depthwise Separable Conv2D layers
- The model uses ReLU activation function and Batch Normalization after each convolutional layer.
- Total number of parameters in the model is 148,794.
- Final RF of the model is 47.

## Requirements

python
torch
torchvision
albumentations
numpy
tqdm


## Model Performance
- Target Accuracy: 85%
- Training Time: Varies by hardware
- Validation performed after each epoch

## Dataset
CIFAR-10:
- 50,000 training images
- 10,000 testing images
- 10 classes
- 32x32 RGB images

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

# Receptive Field Calculation

Input: 1x1
Conv1 (3x3): 3x3
Conv1 (5x5): 7x7
Conv2 (3x3): 9x9
Conv2 (1x1): 9x9
Conv2 (3x3): 11x11
Conv3 (3x3, dilation=4): 27x27
Conv4 (3x3, stride=2): 47x47

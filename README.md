# **FashionMNIST Classification**

This project implements and compares three neural network architectures for classifying the FashionMNIST dataset. Models are trained, evaluated, and visually compared for accuracy and performance.

---

## **Dataset**

- **Dataset**: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), a collection of 28x28 grayscale images of 10 clothing categories.
- **Classes**: `['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`
<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/d59de12b-775f-4232-abab-4596978b8ec1" width="55%" />
  <br />
  <strong>Data used</strong>
</p>

---

## **Model Architectures**

### **Model 0: Basic Linear Layers**
- **Layers**: 
  - `Flatten`
  - `Linear(784 → 10)`
- **Activation**: None
- **Purpose**: A simple baseline to observe classification results with minimal computation.

---

### **Model 1: Linear Layers with Activation**
- **Layers**: 
  - `Flatten`
  - `Linear(784 → 10)` → `ReLU`
  - `Linear(10 → 10)` → `ReLU`
- **Activation**: `ReLU`
- **Purpose**: Adding non-linearity to improve classification.

---

### **Model 2: TinyVGG-like CNN**
- **Layers**:
  - `Conv2d(1 → 10, kernel_size=3, padding=1)` → `ReLU`
  - `Conv2d(10 → 10, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(kernel_size=2)`
  - `Conv2d(10 → 10, kernel_size=3, padding=1)` → `ReLU`
  - `Conv2d(10 → 10, kernel_size=3, padding=1)` → `ReLU` → `MaxPool2d(kernel_size=2)`
  - `Linear(10*7*7 → 10)`
- **Activation**: `ReLU`
- **Purpose**: Leveraging convolutional layers for feature extraction and classification.

---

## **Training Setup**

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `SGD` (Learning rate: 0.1)
- **Batch Size**: 32
- **Epochs**: 3
- **Device**: GPU (`cuda`) if available, otherwise CPU

---

## **Results**

<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/da10679e-3944-43b6-8139-985b60028f92" width="55%" />
  <br />
  <strong>Model performances</strong>
</p>

- **Model 2 outperforms others** due to its convolutional architecture.

<p float="left" style="text-align: center; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/918a0ed3-8b11-4626-b177-6ab46953ca97" width="55%" />
  <br />
  <strong>Output pre-trained model</strong>
</p>

---


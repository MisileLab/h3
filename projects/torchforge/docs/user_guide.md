# TorchForge User Guide

## Introduction

TorchForge is a visual PyTorch model builder that allows you to design, train, and export neural networks using an intuitive drag-and-drop interface. This guide will help you get started with building your first models.

## Quick Start

### Installation

```bash
pip install torchforge
```

### Launching the Application

```python
import torchforge
torchforge.run()
```

## Building Your First Model

### 1. Understanding the Interface

The main window consists of several key areas:

- **Toolbox (Left)**: Contains available layer nodes that you can drag onto the canvas
- **Graph Canvas (Center)**: The main area where you build your neural network
- **Properties Panel (Right)**: Displays and allows editing of selected node properties
- **Training Panel (Bottom)**: Configure and monitor model training
- **Experiments Panel (Optional)**: Manage and compare different experiments

### 2. Creating a Simple CNN

Let's create a simple Convolutional Neural Network for MNIST digit classification:

#### Step 1: Add Layers

1. Click on **Conv2D** in the toolbox to add a convolutional layer
2. Click on **ReLU** to add an activation function
3. Click on **MaxPool2D** to add a pooling layer
4. Click on **Flatten** to flatten the feature maps
5. Click on **Linear** to add a fully connected layer
6. Add another **Linear** layer for the final output

#### Step 2: Configure Layers

1. Select the first Conv2D layer and set its properties:
   - in_channels: 1 (MNIST images are grayscale)
   - out_channels: 32
   - kernel_size: 3

2. Select the first Linear layer:
   - in_features: 32 * 13 * 13 (calculated from previous layers)
   - out_features: 128

3. Select the final Linear layer:
   - in_features: 128
   - out_features: 10 (10 digits)

#### Step 3: Connect Layers

Click and drag from the output port (right side) of one layer to the input port (left side) of the next layer to connect them.

#### Step 4: Validate the Model

Press **F5** or use **Model → Validate Graph** to check if your model is valid.

### 3. Training the Model

#### Step 1: Configure Training

1. Go to the **Training Panel**
2. Set the following parameters:
   - Dataset: MNIST
   - Batch Size: 32
   - Epochs: 10
   - Learning Rate: 0.001
   - Optimizer: Adam
   - Loss Function: CrossEntropyLoss

#### Step 2: Start Training

Click the **Start Training** button to begin training. You can monitor the progress in real-time:

- Loss and accuracy curves
- Training metrics
- Console output with detailed information

#### Step 3: Monitor Progress

The training panel shows:
- Current epoch and progress bar
- Training and validation loss
- Training and validation accuracy
- Real-time plots of metrics

## Available Layers

### Convolutional Layers

#### Conv2D
2D Convolution layer for image data.

**Properties:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of the convolution kernel
- `stride`: Stride of the convolution
- `padding`: Zero-padding added to both sides
- `dilation`: Spacing between kernel elements
- `groups`: Number of blocked connections
- `bias`: Whether to include a bias term

#### MaxPool2D
2D Max pooling layer for downsampling.

**Properties:**
- `kernel_size`: Size of the pooling window
- `stride`: Stride of the pooling
- `padding`: Padding added to both sides
- `dilation`: Spacing between kernel elements
- `return_indices`: Whether to return indices
- `ceil_mode`: Use ceil instead of floor for output shape

### Linear Layers

#### Linear
Fully connected layer.

**Properties:**
- `in_features`: Number of input features
- `out_features`: Number of output features
- `bias`: Whether to include a bias term

### Activation Functions

#### ReLU
Rectified Linear Unit activation.

**Properties:**
- `inplace`: Whether to modify input in-place

#### Sigmoid
Sigmoid activation function.

No configurable properties.

### Utility Layers

#### Flatten
Flattens a contiguous range of dimensions.

**Properties:**
- `start_dim`: First dimension to flatten
- `end_dim`: Last dimension to flatten

## Training Configuration

### Datasets

TorchForge supports several built-in datasets:

- **MNIST**: Handwritten digits (28x28 grayscale, 10 classes)
- **Fashion-MNIST**: Fashion items (28x28 grayscale, 10 classes)
- **CIFAR-10**: Color images (32x32 RGB, 10 classes)

### Optimizers

- **Adam**: Adaptive Moment Estimation
- **SGD**: Stochastic Gradient Descent
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square Propagation

### Loss Functions

- **CrossEntropyLoss**: For multi-class classification
- **MSELoss**: Mean Squared Error for regression
- **BCELoss**: Binary Cross Entropy for binary classification

## Exporting Models

### Export as PyTorch Code

1. Use **File → Export Code** or press **Ctrl+E**
2. Choose a location to save the Python file
3. The generated code includes:
   - Model class definition
   - Layer initialization
   - Forward pass implementation
   - Example usage

### Export Trained Model

1. After training, use **File → Export Model**
2. Save the model weights (.pth file)
3. You can load this model later using:
   ```python
   import torch
   model = torch.load('your_model.pth')
   ```

## Experiment Management

### Creating Experiments

1. Open the **Experiments Panel** (View → Experiments)
2. Click **New** to create a new experiment
3. Configure your model and training parameters
4. Start training - the experiment will be automatically saved

### Comparing Experiments

1. Select multiple experiments in the list
2. Click **Compare** to view side-by-side comparisons
3. Compare:
   - Hyperparameters
   - Training curves
   - Final performance metrics

## Tips and Best Practices

### Model Design

1. **Start Simple**: Begin with a simple architecture and gradually add complexity
2. **Check Dimensions**: Use the validation feature to ensure layer dimensions match
3. **Monitor Parameters**: Keep an eye on the parameter count to avoid overfitting

### Training

1. **Learning Rate**: Start with 0.001 and adjust based on training behavior
2. **Batch Size**: Use 32-64 for most cases, larger if memory allows
3. **Early Stopping**: Monitor validation loss to prevent overfitting

### Performance

1. **GPU Acceleration**: Enable CUDA in the device settings if available
2. **Data Loading**: Use appropriate batch sizes for efficient GPU utilization
3. **Model Size**: Consider model complexity for your target deployment platform

## Troubleshooting

### Common Issues

#### "Graph contains cycles"
- Check for circular connections in your model
- Ensure data flows from input to output without loops

#### "Dimension mismatch"
- Verify layer input/output dimensions are compatible
- Use Flatten layers when transitioning from conv to linear layers

#### "Training not starting"
- Ensure your graph is valid (use validation)
- Check that all required properties are set
- Verify dataset is available and properly configured

#### "Poor performance"
- Try different learning rates
- Adjust model architecture
- Increase training epochs
- Check for overfitting/underfitting

### Getting Help

- Check the [GitHub Issues](https://github.com/torchforge/torchforge/issues)
- Review the [Developer Guide](developer_guide.md)
- Join our community discussions

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Project | Ctrl+N |
| Open Project | Ctrl+O |
| Save Project | Ctrl+S |
| Export Code | Ctrl+E |
| Validate Graph | F5 |
| Generate Code | F6 |
| Start Training | F9 |
| Stop Training | F10 |
| Delete Selected | Del |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |

## Advanced Features

### Custom Datasets

While TorchForge includes built-in datasets, you can use custom image datasets by organizing them in the following structure:

```
custom_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    └── class2/
```

### Model Checkpointing

Enable checkpoint saving in the training configuration to:
- Resume training from interruptions
- Save the best model during training
- Track model evolution over time

### Hyperparameter Tuning

Use the experiment management system to:
- Compare different hyperparameter settings
- Track performance across multiple runs
- Identify optimal configurations

---

For more detailed technical information, see the [Developer Guide](developer_guide.md).
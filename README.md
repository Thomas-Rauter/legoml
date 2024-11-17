# LegoML

LegoML is a modular library for machine learning workflows. Inspired by Lego 
blocks, it aims to combine maximum usability with flexibility, enabling you to
quickly build robust solutions.

## Features
- Generalized training loops
- Modular visualization utilities
- Easy integration with PyTorch
- Extensible for advanced tasks

## Installation
Install directly from GitHub:
```bash
pip install git+https://github.com/yourusername/legoml.git
```

# Usage

## Training

```python
from legoml.training.loops import train_model
from legoml.models.cnn import CNNModel

model = CNNModel(...)
train_model(model, train_loader, val_loader, ...)
```

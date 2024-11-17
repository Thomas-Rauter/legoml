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
pip install git+https://github.com/Thomas-Rauter/legoml.git@0.1.0
```

Install into a Jupyter notebook:
```bash
!pip install git+https://github.com/Thomas-Rauter/legoml.git@0.1.0
```

# Usage

## Training

```python
from legoml.training.loops import train_model
from legoml.models.cnn import CNNModel

model = CNNModel(...)
train_model(model, train_loader, val_loader, ...)
```

# Philosophy of LegoML
LegoML is designed with a simple yet powerful philosophy: to provide modular,
simple, and flexible building blocks for machine learning workflows.
Much like LEGO bricks, these components can be easily combined and customized
to build solutions tailored to your specific needs.

### 1. Modularity
- **Core Principle**: Modularity is at the heart of LegoML. Each function or class is a small, self-contained building block that serves a specific purpose.
- **Why It Matters**: Modularity allows you to mix, match, and replace components as your project evolves, ensuring that your workflow remains adaptable.
- **Example**: Instead of bundling a rigid, monolithic pipeline, LegoML offers standalone pieces like a `train_one_epoch` function or a `visualize_metrics` utility that you can plug into your own training loops or pipelines.

---

### 2. Simplicity
- **Core Principle**: Simplicity is essential. LegoML functions should be intuitive and require minimal setup to get started.
- **Why It Matters**: If a building block requires excessive effort to understand or use, you might as well write it yourself. LegoML is designed to save you time, not create additional complexity.
- **Example**: A general-purpose `train_loop` function abstracts away the boilerplate code for training a PyTorch model while remaining simple enough for anyone familiar with PyTorch to understand.

---

### 3. Flexibility
- **Core Principle**: Flexibility ensures that the building blocks are adaptable to a wide range of use cases.
- **Why It Matters**: Special-purpose, overly specific functions often fail to accommodate unique scenarios. LegoML avoids this by prioritizing generality, enabling its components to cover most common use cases in machine learning workflows.
- **Example**: A single training loop function can accommodate any PyTorch model, optimizer, or learning rate scheduler, instead of being tied to a particular architecture or configuration.

---

### Why This Matters
LegoML doesn’t aim to reinvent the wheel; it aims to make your work more
efficient. By focusing on modularity, simplicity, and flexibility, 
it empowers you to focus on solving data science problems rather than 
spending excessive time on repetitive coding tasks. Whether you're a beginner
or a seasoned practitioner, LegoML provides the tools you need to build robust
machine learning solutions, brick by brick.
import torch
from typing import Union


def train_torch_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Union[callable, torch.nn.Module],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
) -> float:
    """
    Performs one training epoch of a PyTorch neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_loader : torch.utils.data.DataLoader
        The dataLoader for the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    criterion : callable
        A loss function or custom criterion callable. It must accept two inputs:
        - outputs: A tensor representing the model's predictions, typically of
          shape (batch_size, ...) depending on the task.
        - targets: A tensor representing the true labels or values, typically of
          shape (batch_size, ...) matching the outputs' expected dimensions.

        The callable must return:
        - A scalar loss value (as a tensor) that represents the computed loss
          for the batch.
    device : torch.device, optional
        Device to run training on. Default is GPU if available, otherwise CPU.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    if next(model.parameters()).device != device:
        model = model.to(device)

    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:

        if inputs.device != device:
            inputs = inputs.to(device)
        if targets.device != device:
            targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate_torch_epoch(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: Union[callable, torch.nn.Module],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
) -> float:
    """
    Performs one validation epoch of a PyTorch neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to validate.
    val_loader : torch.utils.data.DataLoader
        The dataLoader for the validation dataset.
    criterion : callable
        A loss function or custom criterion callable. It must accept two inputs:
        - outputs: A tensor representing the model's predictions, typically of
          shape (batch_size, ...) depending on the task.
        - targets: A tensor representing the true labels or values, typically of
          shape (batch_size, ...) matching the outputs' expected dimensions.

        The callable must return:
        - A scalar loss value (as a tensor) that represents the computed loss
          for the batch.
    device : torch.device, optional
        Device to run validation on. Default is GPU if available, otherwise CPU.

    Returns
    -------
    float
        Average validation loss for the epoch.
    """
    if next(model.parameters()).device != device:
        model = model.to(device)

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:

            if inputs.device != device:
                inputs = inputs.to(device)
            if targets.device != device:
                targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    return running_loss / len(val_loader)

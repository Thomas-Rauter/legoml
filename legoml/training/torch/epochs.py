import torch


def train_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device
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
    criterion : torch.nn.Module
        Loss function (e.g., BCEWithLogitsLoss, CrossEntropyLoss).
    device : torch.device
        Device to run training on (e.g., 'cuda' or 'cpu').

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
) -> float:
    """
    Performs one validation epoch of a PyTorch neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to validate.
    val_loader : torch.utils.data.DataLoader
        The dataLoader for the validation dataset.
    criterion : torch.nn.Module
        Loss function (e.g., BCEWithLogitsLoss, CrossEntropyLoss).
    device : torch.device
        Device to run validation on (e.g., 'cuda' or 'cpu').

    Returns
    -------
    float
        Average validation loss for the epoch.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(val_loader)

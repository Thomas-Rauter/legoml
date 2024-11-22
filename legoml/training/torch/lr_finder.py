import torch
from torch_lr_finder import LRFinder
from typing import Union


def find_learning_rate_torch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Union[callable, torch.nn.Module],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
        val_loader: torch.utils.data.DataLoader = None,
        plot: bool = True,
) -> float:
    """
    Applies a learning rate range test to suggest an initial learning rate.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate.
    train_loader : torch.utils.data.DataLoader
        The dataLoader for training data.
    optimizer : torch.optim.Optimizer
        Pre-configured optimizer for the model parameters.
    criterion : callable
        Loss function to evaluate during the test (e.g., nn.CrossEntropyLoss).
    device : torch.device, optional
        Device to run the test on. Default is GPU if available, otherwise CPU.
    start_lr : float, optional
        Starting learning rate for the test. Default is 1e-7.
    end_lr : float, optional
        Maximum learning rate for the test. Default is 10.
    num_iter : int, optional
        Number of iterations for the range test. Default is 100.
    step_mode : str, optional
        Mode for learning rate increase: "exp" (exponential) or "linear".
        Default is "exp".
    val_loader : torch.utils.data.DataLoader, optional
        Validation DataLoader for evaluation during the test.
    plot : bool, optional
        Whether to plot the loss-learning rate graph. Default is True.

    Returns
    -------
    float
        Suggested learning rate based on the test results.
    """
    if next(model.parameters()).device != device:
        model = model.to(device)

    # Initialize LR Finder
    lr_finder = LRFinder(
        model,
        optimizer,
        criterion,
        device=device
    )

    # Perform the range test
    lr_finder.range_test(
        train_loader=train_loader,
        val_loader=val_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
    )

    # Optionally plot the results
    if plot:
        lr_finder.plot()

    # Reset model and optimizer to initial state
    lr_finder.reset()

    # Optionally return a suggested learning rate
    losses = lr_finder.history["loss"]
    lrs = lr_finder.history["lr"]
    min_loss_idx = min(range(len(losses)), key=losses.__getitem__)
    suggested_lr = lrs[min_loss_idx]
    print(f"Suggested learning rate: {suggested_lr:.2e}")

    return suggested_lr

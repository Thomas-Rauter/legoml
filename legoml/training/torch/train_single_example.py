# External imports
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Union

# Internal imports
from legoml.training.torch.epochs import train_torch_epoch


def train_single_torch_example(
        model: torch.nn.Module,
        example_input: torch.Tensor,
        example_target: torch.Tensor,
        criterion: Union[callable, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        num_steps: int = 100,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
) -> list:
    """
    Train a PyTorch model on a single example for debugging,
     using train_torch_epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    example_input : torch.Tensor
        The input tensor for the single example.
    example_target : torch.Tensor
        The target tensor for the single example.
    criterion : Union[callable, torch.nn.Module]
        Loss function for the task.
    optimizer : torch.optim.Optimizer
        Pre-configured optimizer for the model parameters.
    num_steps : int, optional
        Number of gradient steps for the single-example training.
        Default is 100.
    device : torch.device, optional
        Device to run training on. Default is GPU if available, otherwise CPU.

    Returns
    -------
    list
        A list of average losses recorded for each epoch.
    """
    # Create a DataLoader for the single example
    single_example_dataset = TensorDataset(
        example_input,
        example_target
    )
    single_example_loader = DataLoader(
        single_example_dataset,
        batch_size=1
    )

    # List to store losses for each step/epoch
    losses = []

    for step in range(num_steps):
        # Train one epoch (one step in this case)
        epoch_loss = train_torch_epoch(
            model=model,
            train_loader=single_example_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        losses.append(epoch_loss)

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {epoch_loss:.4f}")

    return losses

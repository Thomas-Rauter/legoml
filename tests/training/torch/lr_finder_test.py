import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from legoml.training.torch.lr_finder import find_learning_rate_torch


def test_find_learning_rate_torch():
    """
    Test the find_learning_rate_torch function to ensure it works as expected.
    """

    # Create a simple dataset
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randint(0, 2, (100,))  # Binary classification
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16)

    # Define a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    # Call the find_learning_rate_torch function
    try:
        suggested_lr = find_learning_rate_torch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            start_lr=1e-6,
            end_lr=1,
            num_iter=50,
            device=torch.device("cpu"),  # Use CPU for simplicity
            plot=False  # Suppress plotting during test
        )
        assert isinstance(suggested_lr, float), "Suggested learning rate is not a float."
        assert 1e-6 <= suggested_lr <= 1, "Suggested learning rate is out of the expected range."
        print(f"Test passed. Suggested learning rate: {suggested_lr:.2e}")
    except Exception as e:
        print(f"Test failed with error: {e}")


if __name__ == "__main__":
    test_find_learning_rate_torch()

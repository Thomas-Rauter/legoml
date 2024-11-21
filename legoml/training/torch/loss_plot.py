import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class LiveLossPlot:
    """
    A class for creating a live loss plot that updates dynamically
    during training.

    Parameters
    ----------
    val_loss : bool, optional
        Whether to include validation loss in the plot. Default is True.
    in_notebook : bool, optional
        Whether the script is running in a Jupyter notebook. Default is True.
    """

    def __init__(self, val_loss: bool = True, in_notebook: bool = True):
        self.train_losses = []
        self.val_losses = []
        self.val_loss_enabled = val_loss
        self.in_notebook = in_notebook
        self.fig, self.ax = plt.subplots()
        if in_notebook:
            display(self.fig)

    def update(self, train_loss: float, val_loss: float = None):
        """
        Updates the plot with new training (and optional validation)
         loss values.

        Parameters
        ----------
        train_loss : float
            The training loss value for the current epoch.
        val_loss : float, optional
            The validation loss value for the current epoch. Only used if
            val_loss is enabled.
        """
        self.train_losses.append(train_loss)
        if self.val_loss_enabled and val_loss is not None:
            self.val_losses.append(val_loss)

        self._draw_plot()

    def _draw_plot(self):
        """Draws or updates the plot with the current loss values."""
        self.ax.clear()
        epochs = range(1, len(self.train_losses) + 1)
        self.ax.plot(epochs, self.train_losses, label='Train Loss', marker='o')
        if self.val_loss_enabled and self.val_losses:
            self.ax.plot(epochs, self.val_losses, label='Validation Loss',
                         marker='o')

        self.ax.set_title('Loss Over Epochs')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True)

        if self.in_notebook:
            clear_output(wait=True)
            display(self.fig)
        else:
            plt.pause(0.01)

    def close(self):
        """Closes the plot to release resources."""
        plt.close(self.fig)

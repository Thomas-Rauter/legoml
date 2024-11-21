import time
from legoml.plots.loss_plot import LiveLossPlot


def test_live_loss_plot_live_update():
    """
    Integration test for the LiveLossPlot class to showcase live updates.
    This test displays a live plot updating dynamically with mock data.
    """
    # Mock data
    train_losses = [0.8, 0.6, 0.4, 0.3]
    val_losses = [0.7, 0.5, 0.35, 0.25]

    # Initialize the LiveLossPlot
    plotter = LiveLossPlot(val_loss=True, in_notebook=False)

    try:
        for train_loss, val_loss in zip(train_losses, val_losses):
            plotter.update(train_loss=train_loss, val_loss=val_loss)
            time.sleep(1)  # Simulate delay between epochs
    finally:
        plotter.close()  # Ensure resources are cleaned up


test_live_loss_plot_live_update()

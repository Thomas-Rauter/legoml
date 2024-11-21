import tensorflow as tf
from typing import Union


def train_tf_epoch(
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        criterion: Union[callable, tf.keras.losses.Loss]
) -> float:
    """
    Performs one training epoch of a TensorFlow Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow Keras model to train.
    train_dataset : tf.data.Dataset
        The dataset for the training data.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer for updating model parameters.
    criterion : Union[callable, tf.keras.losses.Loss]
        A loss function or custom criterion callable. It must accept two inputs:
        - targets: A tensor representing the true labels or values, typically of
          shape (batch_size, ...) depending on the task.
        - predictions: A tensor representing the model's predictions, typically
          of shape (batch_size, ...) matching the expected dimensions.

        The callable must return:
        - A scalar loss value (as a tensor) that represents the computed loss
          for the batch.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    running_loss = 0.0
    num_batches = 0

    for inputs, targets in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = criterion(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        running_loss += loss.numpy()
        num_batches += 1

    return running_loss / num_batches


def validate_tf_epoch(
        model: tf.keras.Model,
        val_dataset: tf.data.Dataset,
        criterion: Union[callable, tf.keras.losses.Loss]
) -> float:
    """
    Performs one validation epoch of a TensorFlow Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow Keras model to validate.
    val_dataset : tf.data.Dataset
        The dataset for the validation data.
    criterion : Union[callable, tf.keras.losses.Loss]
        A loss function or custom criterion callable. It must accept two inputs:
        - targets: A tensor representing the true labels or values, typically of
          shape (batch_size, ...) depending on the task.
        - predictions: A tensor representing the model's predictions, typically
          of shape (batch_size, ...) matching the expected dimensions.

        The callable must return:
        - A scalar loss value (as a tensor) that represents the computed loss
          for the batch.

    Returns
    -------
    float
        Average validation loss for the epoch.
    """
    running_loss = 0.0
    num_batches = 0

    for inputs, targets in val_dataset:
        predictions = model(inputs, training=False)
        loss = criterion(targets, predictions)
        running_loss += loss.numpy()
        num_batches += 1

    return running_loss / num_batches

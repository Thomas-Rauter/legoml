import random
import numpy as np
import torch
import tensorflow as tf


def set_torch_random_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    cuda: bool = True,
    numpy_seed: bool = True,
    random_seed: bool = True
) -> None:
    """
    Sets random seeds for reproducibility across libraries and allows
    customization of various settings for deterministic computation.

    Parameters
    ----------
    seed : int, optional
        The seed value to use for random number generators. Default is 42.
    deterministic : bool, optional
        Whether to enable deterministic behavior. When True,
        ensures reproducible behavior by disabling CuDNN non-deterministic
        algorithms. Default is True.
    benchmark : bool, optional
        Whether to enable CuDNN benchmark mode. When True, allows CuDNN to
        select the fastest algorithm based on the hardware, which may lead to
        non-deterministic results. Default is False.
    cuda : bool, optional
        Whether to apply seed settings to CUDA (GPU) computations.
        Default is True.
    numpy_seed : bool, optional
        Whether to seed NumPy's random number generator. Default is True.
    random_seed : bool, optional
        Whether to seed Python's built-in random module. Default is True.

    Returns
    -------
    None
    """
    if random_seed:
        random.seed(seed)

    if numpy_seed:
        np.random.seed(seed)

    torch.manual_seed(seed)

    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = benchmark


def set_tf_random_seed(
    seed: int = 42,
    deterministic: bool = True,
    numpy_seed: bool = True,
    random_seed: bool = True
) -> None:
    """
    Sets random seeds for reproducibility across TensorFlow and other libraries,
    with options for deterministic computation.

    Parameters
    ----------
    seed : int, optional
        The seed value to use for random number generators. Default is 42.
    deterministic : bool, optional
        Whether to enable deterministic behavior in TensorFlow operations.
        When True, ensures reproducible behavior by disabling TensorFlow's
        non-deterministic algorithms. Default is True.
    numpy_seed : bool, optional
        Whether to seed NumPy's random number generator. Default is True.
    random_seed : bool, optional
        Whether to seed Python's built-in random module. Default is True.

    Returns
    -------
    None
    """
    if random_seed:
        random.seed(seed)

    if numpy_seed:
        np.random.seed(seed)

    tf.random.set_seed(seed)

    if deterministic:
        # Enforce deterministic operations
        # (only effective for certain versions of TensorFlow)
        tf.config.experimental.enable_op_determinism()

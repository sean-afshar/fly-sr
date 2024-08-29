# Imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

n_pixels = 96  # Number of partitions for angular discretization


def to_theta(x: ArrayLike) -> ArrayLike:
    """Converts 1-hot encoded vector to an angle.

    Angles are taken by discretizing [0, 360) into x.shape[-1] even bins and
    taking the lower bound of the bin corresponding to the index where x = 1.

    Args:
      x: Array of shape (..., n_partitions), vectors must be 1-hot encoded.

    Returns:
      theta: Array of shape (...,).
    """
    return jnp.argmax(x, axis=-1) * (360 / x.shape[-1])


def to_1_hot(theta: ArrayLike, n_partitions: int = 96) -> jax.Array:
    """Converts an angle to a 1 hot encoding.

    Args:
      theta: angle in [0, 360)
      n_paritions: How many even sized bins to discretize [0, 360) into

    Returns:
      x: 1 hot vector of shape (n_partitions,)
    """
    return jax.nn.one_hot((theta % 360) // (360 / n_partitions), n_partitions)


def action_pdf(heading: ArrayLike, goal: ArrayLike, phi: ArrayLike, psi: ArrayLike) -> ArrayLike:
    """Give a probability distribution over

    Args:
        action (int): Index corresponding to the action taken (0 for fixations, 1 for left saccades, 2 for right saccades)
        heading (ArrayLike): 1 hot vector
        goal (ArrayLike): 1 hot vector
        phi (ArrayLike): Tensor of shape |A| x D x len(heading/goal)
        psi (ArrayLike): Tensor of shape D x len(heading/goal)

    Returns:
        float: P(action | goal, heading) for each action
    """
    logits = np.einsum("ijk,k,jl,l->i", phi, heading, psi, goal)
    return jax.nn.softmax(logits)

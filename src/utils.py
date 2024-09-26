# Imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax

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
  

def contrastive_loss(
    params: optax.Params,
    anchor_states: ArrayLike,
    like_states: ArrayLike,
    dislike_states: ArrayLike,
):
    """Computes contrastive loss between a single anchor, like and dislike states.

    L(s, s+, s-) = E[log(sigmoid(<phi^T s, psi^T s+ >)) + log(1 - sigmoid(<phi^T s,  psi^T s- >))]

    Args:
        params (optax.Params): Dictionary containing a 'phi' parameter of shape (latent_dim, anchor_dim)
          and 'psi' parameter of shape (latent_dim, dim)
        anchor_state (ArrayLike): Array of shape (batch, anchor_dim)
        like_state (ArrayLike): Array of shape (batch, dim)
        dislike_state (ArrayLike): Array of shape (batch, dim)

    Returns:
        loss (float): Binary cross entropy loss
    """
    # <(phi^T s[i]), (psi^T s+[i])>
    like_logits = jnp.einsum(
        "ij,kj,il,kl->i", anchor_states, params["phi"], like_states, params["psi"]
    )  # (batch,)
    # <(phi^T s[i]), (psi^T s-[i])>
    dislike_logits = jnp.einsum(
        "ij,kj,il,kl->i", anchor_states, params["phi"], dislike_states, params["psi"]
    )  # (batch,)
    # sigma(-x) = 1 - sigma(x), more numerically stable
    minibatch_loss = -(jax.nn.log_sigmoid(like_logits) + jax.nn.log_sigmoid(-dislike_logits))  # (batch,)
    return jnp.mean(minibatch_loss)
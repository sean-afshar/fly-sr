import jax
import jax.numpy as jnp
from jax import vmap
from jax.typing import ArrayLike
from typing import Tuple, Optional, Callable
from tensorflow_probability.substrates import jax as tfp

n_pixels = 96  # Number of partitions for angular discretization


def sample_direction(key: jax.Array) -> ArrayLike:
    """Generates samples of 1 hot encoded goal vectors.

    Args:
      key: Random key for sampling

    Returns:
      x: float in [0, 360)
    """
    return jax.random.uniform(key, minval=0.0, maxval=360.0)


def sample_action(key: jax.Array, n_actions: Optional[int] = 3, p: Optional[ArrayLike] = None) -> int:
    """Samples an action based on the given key and probability distribution.

    Args:
      key (jax.Array): The random key used for sampling.
      n_actions: Size of action space, default 3.
      p (Optional): Discrete probability over action space, assumed to be uniform.

    Returns:
      action: Index corresponding to element in action space, default 0 for fixation, 1 for left saccade, 2 for right saccade.
    """
    return jax.random.choice(key, jnp.arange(n_actions), p=p)


def sample_saccade(
    key: jax.Array,
    direction: int,
    mu_s: Optional[float] = 3.89,
    sigma_s: Optional[float] = 0.54,
    a_dt: Optional[float] = 0.56,
    eta_dt: Optional[float] = 1.0,
    f_min: Optional[float] = -1.26,
    f_max: Optional[float] = 2.33,
    f_0: Optional[float] = 0.35,
    slope: Optional[float] = 7.55,
) -> Tuple[float, float]:
    """Samples saccade velocity and duration.

    The velocity magnitudes are drawn from a lognormal distribution v ~ lognormal(mu_s, sigma_s**2).
    The time length of the saccade will depend on the velocity. The time lengths are drawn from an inverse
    gaussian like the fixations, which has an explicity velocity dependence, dt ~ IG(f(velocity), (a_dt / eta_dt)**2).
    f(velocity) = f_max / (1 + exp(-slope(velocity - f_0))) + f_min.

    Args:
        key (jax.Array): Random key for sampling.
        direction (int): +1 for CCW rotations (left) and -1 for CW rotations (right)
        mu_s (Optional[float], optional): Mean for lognormal distribution over velocity magnitudes. Defaults to 3.89.
        sigma_s (Optional[float], optional): Standard dev for lognormal distribution over velocity magnitudes. Defaults to 0.54.
        a_dt (Optional[float], optional): Boundary for drift diffusion process underlying saccade times. Defaults to 0.56.
        eta_dt (Optional[float], optional): Dispersion parameter for drift diffusion process. Defaults to 1.0.
        f_min (Optional[float], optional): Min value of . Defaults to -1.26.
        f_max (Optional[float], optional): Max value of . Defaults to 2.33.
        f_0 (Optional[float], optional): Intercept. Defaults to 0.35.
        slope (Optional[float], optional): Slope of sigmoid. Defaults to 7.55.

    Returns:
        Tuple[float, float]: A tuple containing the turning velocity and sampled saccade duration.
    """
    mod_key, dt_key = jax.random.split(key, 2)
    # Sample angular velocity
    modulus = tfp.distributions.LogNormal(loc=mu_s, scale=sigma_s).sample(seed=mod_key)
    # Sample saccade duration
    mu_dt = a_dt / (f_max * jax.nn.sigmoid(slope * (modulus - f_0)) - f_min)
    dt = tfp.distributions.InverseGaussian(loc=mu_dt, concentration=(a_dt / eta_dt) ** 2).sample(seed=dt_key)
    return direction * modulus, dt


def sample_fixation(
    key: jax.Array, mu_f: float = 1.0, a_f: float = 0.79, eta_f: float = 1.0
) -> Tuple[float, float]:
    """
    Samples fixation parameters for a given action.

    Fixation lengths are governed by an underlying drift diffusion process, resulting in an inverse gaussian distribution.
    dt ~ InverseGaussian(mu_f, (a_f / eta_f)^2)

    Args:
        key (jax.Array): Random key for sampling.
        mu_f (float, optional): Mean for the inverse Gaussian distribution over fixation durations. Defaults to 1.0.
        a_f (float, optional): Boundary for the drift diffusion process underlying fixation times. Defaults to 0.79.
        eta_f (float, optional): Dispersion parameter for the drift diffusion process. Defaults to 1.0.

    Returns:
        Tuple[float, float]: A tuple containing the fixation velocity (always 0.0) and the sampled fixation duration.
    """
    sample_key = jax.random.split(key, 1)
    return 0.0, tfp.distributions.InverseGaussian(loc=mu_f, concentration=(a_f / eta_f) ** 2).sample(
        seed=sample_key
    )


def sample_action_params(
    key: jax.Array,
    action: int,
    mu_s: Optional[float] = 3.89,
    sigma_s: Optional[float] = 0.54,
    a_dt: Optional[float] = 0.56,
    eta_dt: Optional[float] = 1.0,
    f_min: Optional[float] = -1.26,
    f_max: Optional[float] = 2.33,
    f_0: Optional[float] = 0.35,
    slope: Optional[float] = 7.55,
    mu_f: float = 1.0,
    a_f: float = 0.79,
    eta_f: float = 1.0,
) -> Tuple[float, float]:
    """Given an action, generates angular velocity and time length.

    Args:
        key (jax.Array): Key for sampling.
        action (int): Assumed to be 0 = fixation, 1 = left saccade, 2 = right saccade
        **kwargs: see sample_fixation and sample_saccade for more

    Returns:
        Tuple[float, float]: A tuple containing the turning veloicty and sampled time length.
    """
    # Convert action into direction
    direction = (-2 * action) + 3
    # Sample
    velocity, dt = jax.lax.cond(
        action == 0,
        lambda k: sample_fixation(k, mu_f, a_f, eta_f),
        lambda k: sample_saccade(k, direction, mu_s, sigma_s, a_dt, eta_dt, f_min, f_max, f_0, slope),
        key,
    )
    return velocity, dt


def generate_trajectory(
    key: jax.Array,
    horizon: int,
    action_sampler: Callable = sample_action,
    parameter_sampler: Callable = sample_action_params,
) -> Tuple[ArrayLike, ArrayLike]:
    """Generates a trajectory based on a given random key and horizon.

    Args:
        key (jax.Array): Random key for sampling.
        horizon (int): The number of steps in the trajectory.
        action_sampler (Callable, optional): Function to sample actions. Function signature should be f(key) -> int.
        parameter_sampler (Callable, optional): Function to sample action parameters. Should have signature f(sampling_key, action) -> Tuple[...]. Defaults to sample_action_params.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The generated trajectory and the goal direction. The generated trajectory is a length horizon list of (heading, ang_velocity, dt) tuples.
    """
    # Initialize conditions
    init_keys = jax.random.split(key, 4)
    x_0, x_G = vmap(sample_direction)(init_keys[:2])
    a_0 = action_sampler(init_keys[2])
    dtheta_0, dt_0 = parameter_sampler(init_keys[3], a_0)

    def update(carry, key):
        # Integrate angular velocity
        prev_state, dtheta, dt = carry
        state = prev_state + dtheta * dt
        # Sample next action
        action = action_sampler(key)
        # Sample action parameters
        dtheta, dt = parameter_sampler(key, action)
        return (state, dtheta, dt), (state, dtheta, dt)

    # Scan through a trajectory
    traj_keys = jax.random.split(init_keys[3], horizon)
    _, trajectory = jax.lax.scan(update, (x_0, dtheta_0, dt_0), traj_keys, horizon)

    return trajectory, x_G


def sample_trajectory(
    key,
    design_mat: ArrayLike,
    tau_s: int,
    tau_f: int,
) -> Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike], Tuple[int, int]]:
    """Samples like and dislike pairs of states from a trajectory.

    Args:
        key (jax.Array): Random key for sampling.
        design_mat (ArrayLike): Design matrix containing trajectory data.
        tau_s (int): Parameter for sampling time lag during saccades.
        tau_f (int): Parameter for sampling time lag during fixations.

    Returns:
        Tuple of like states (s, s+), tuple of dislike states (s, s-), indices for like and dislike states.
    """

    def sample_like(sample_key, design_mat, traj_index, state_index, tau_s, tau_f):
        # Recognize previous action
        is_fixated = design_mat[traj_index][state_index][1] == 0
        # Sample time lag
        time_step = jax.lax.cond(
            is_fixated,
            lambda k: jax.random.geometric(k, p=tau_f),
            lambda k: jax.random.geometric(k, p=tau_s),
            sample_key,
        )
        # Sample next state
        next_index = jnp.max(jnp.array([len(design_mat[traj_index]) - 1, state_index + time_step]))
        pair = design_mat[traj_index][next_index]
        return pair, next_index * (traj_index + 1)

    def sample_dislike(sample_key, traj_index, design_mat):
        traj_key, state_key = jax.random.split(sample_key)
        candidates = jnp.arange(design_mat.shape[0])
        candidates = candidates[candidates != traj_index]
        traj_index = jax.random.choice(traj_key, candidates)
        state_index = jax.random.choice(state_key, jnp.arange(design_mat.shape[1]))
        return design_mat[traj_index][state_index], state_index * (traj_index + 1)

    # Sample anchor state
    anchor_key, like_key, dislike_key = jax.random.split(key, 3)
    anchor_traj_key, anchor_state_key = jax.random.split(anchor_key)
    anchor_traj_index = jax.random.choice(anchor_traj_key, jnp.arange(design_mat.shape[0]))
    anchor_state_index = jax.random.choice(anchor_state_key, jnp.arange(design_mat.shape[1]))
    anchor_state = design_mat[anchor_traj_index][anchor_state_index]
    # Sample like / dislike pairs
    like_state, like_index = sample_like(
        like_key, anchor_state, anchor_traj_index, anchor_state_index, tau_s, tau_f
    )
    dislike_state, dislike_index = sample_dislike(dislike_key, anchor_traj_index, design_mat)

    return (
        (anchor_state, like_state),
        (anchor_state, dislike_state),
        (like_index, dislike_index),
    )

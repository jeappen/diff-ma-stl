import jax

from ..util import *


class DatasetRolloutGenerator:
    """Parent class for rollout generators that use a dataset"""

    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._num_transitions = self._dataset.obs.shape[0]
        self._batch_size = batch_size

        def _get_batch(data, rng):
            permutation = jax.random.choice(
                rng,
                jnp.arange(self._num_transitions),
                shape=(self._batch_size,),
                replace=False,
            )
            # Sample transitions from dataset
            batch = jtu.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), data
            )
            # Reshape batch to conform with online rollout shape
            return jtu.tree_map(
                lambda x: jnp.reshape(x, (x.shape[0], 1, *x.shape[1:])), batch
            )

        self.batch_fn = jax.jit(_get_batch)

    def batch_rollout(self, rng):
        return self.batch_fn(self._dataset, rng)


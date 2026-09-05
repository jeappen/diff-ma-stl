import jax
import jax.numpy as jnp
import jax.tree_util as jtu

CLOSE_TO_ZERO = 1e-6  # Small value to avoid division by zero


def jax_setdiff1d(a1, a2):
    # Get unique elements from both arrays
    unique_a1 = jnp.unique(a1)
    unique_a2 = jnp.unique(a2)

    # Create broadcasted comparison mask
    mask = ~jnp.any(unique_a1[:, None] == unique_a2, axis=1)

    # Return elements in a1 not present in a2
    return unique_a1[mask]

def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def ema_update(args, denoiser_state, ema_denoiser_state):
    ema_updated_params = jtu.tree_map(
        lambda x, y: args.ema_decay * x + (1 - args.ema_decay) * y,
        ema_denoiser_state.params,
        denoiser_state.params,
    )
    return jtu.tree_map(
        lambda x, y: jnp.where(denoiser_state.step % args.ema_update_every == 0, x, y),
        ema_updated_params,
        ema_denoiser_state.params,
    )


def shuffle_and_batch_dataset(rng, dataset, batch_size):
    """Shuffles and batches dataset (with extra samples truncated)"""
    assert dataset.shape[0] >= batch_size, "Dataset smaller than batch"
    set_shuffled = jax.random.permutation(rng, dataset)
    return set_shuffled[dataset.shape[0] % batch_size:].reshape(
        (-1, batch_size, *dataset.shape[1:])
    )


def merge_vmapped_axes(x):
    """ Only reshape objects with a shape attribute and at least 2 dimensions"""
    if hasattr(x, "shape") and x.ndim >= 2:
        new_shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
        return x.reshape(new_shape)
    return x

import os
import wandb

RESOURCE_DIR = "diffusion-rl/res"
CHECKPOINT_DIR = os.path.join(RESOURCE_DIR, "checkpoints")

LOCAL_CHECKPOINT_DIR = "diff_checkpoints"


def local_checkpoint_config(run_id, base_dir=LOCAL_CHECKPOINT_DIR):
    """Return (config_dict, checkpoint_root) for a self-contained local checkpoint, or None.

    A checkpoint directory is self-contained when it holds the orbax blobs under
    ``CHECKPOINT_DIR`` plus the run's ``config.yaml`` (wandb config format: each key
    nested under ``value``). This lets anyone load a shipped model — e.g. the bundled
    ``diff_checkpoints/qkmvppvt`` — without access to the wandb run it came from;
    the wandb download path is only used as a fallback.
    """
    if not run_id:
        return None
    ckpt_root = os.path.join(base_dir, run_id)
    cfg_path = os.path.join(ckpt_root, "config.yaml")
    if not (os.path.isdir(os.path.join(ckpt_root, CHECKPOINT_DIR)) and os.path.isfile(cfg_path)):
        return None
    import yaml
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    config = {k: v["value"] for k, v in raw.items()
              if isinstance(v, dict) and "value" in v and k != "_wandb"}
    return config, ckpt_root


def log(log_dict):
    wandb.log(log_dict)

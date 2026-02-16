"""Specifically runs the validation loop rollout for a jobconfig."""

import argparse

import torch
import yaml
from tqdm.auto import tqdm

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.rope import ROPEMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType


@torch.no_grad()
def run(cfg_path, debug_n_batches=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    # You can load a ROPE checkpoint into the Pretraining model class because
    # the architecture is the same (only the training implementation is different)
    mdl_class = PretrainingMotionPolicyTransformer
    mdl_path = cfg["load_checkpoint_path"]
    dm = DataModule(
        train_batch_size=cfg["train_batch_size"],
        val_batch_size=cfg["val_batch_size"],
        num_workers=cfg["num_workers"],
        **cfg["data_module_parameters"],
        **cfg["shared_parameters"],
    )
    mdl = mdl_class.load_from_checkpoint(
        mdl_path,
        disable_viz=True,
        **cfg["training_model_parameters"],
        **cfg["shared_parameters"],
    ).to(device)
    dm.setup("fit")
    mdl.setup("fit")
    dl = dm.val_dataloader()[DatasetType.VAL]
    
    # Limit batches in debug mode
    total_batches = len(dl) if debug_n_batches is None else min(debug_n_batches, len(dl))
    
    for ii, batch in enumerate(tqdm(dl, total=total_batches)):
        if debug_n_batches is not None and ii >= debug_n_batches:
            break
        batch = {key: val.to(device) for key, val in batch.items()}
        mdl.trajectory_validation_step(batch, DatasetType.VAL)
    print("Collision Rate:", f"{mdl.val_collision_rate.compute().item():.2%}")
    print("Reaching Success Rate:", f"{mdl.val_reaching_success_rate.compute().item():.2%}")
    print("Success Rate:", f"{mdl.val_success_rate.compute().item():.2%}")
    print(
        "Target Position Error:",
        f"{100 * mdl.val_position_error.compute().item():.2}cm",
    )
    print(
        "Target Orientation Error:",
        f"{mdl.val_orientation_error.compute().item():.2}\N{DEGREE SIGN}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to the configuration YAML file")
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: only process N batches (e.g., --debug 10)",
    )
    args = parser.parse_args()
    run(args.cfg_path, debug_n_batches=args.debug)

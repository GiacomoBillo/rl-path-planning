"""Specifically runs the validation loop rollout for a jobconfig."""

import argparse

import torch
import yaml
from tqdm.auto import tqdm

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.rope import ROPEMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType
from utils.failure_analysis import FailureAnalyzer


@torch.no_grad()
def run(cfg_path, debug_n_batches=None, save_failures=False, failure_output_dir="datasets/failed_trajectories", max_failures=None, abort_on_failure_limit=False):
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
    
    # create failed trajectory analyzer, if enabled
    analyzer = None
    if save_failures:
        dataset_path = cfg.get("data_module_parameters", {}).get("data_dir")
        analyzer = FailureAnalyzer(
            output_dir=failure_output_dir,
            max_failures=max_failures,
            abort_on_limit=abort_on_failure_limit,
            original_dataset_path=dataset_path,
            model_checkpoint=mdl_path,
            model_name=mdl_class.__name__,
        )
    
    kwargs = {
        "disable_viz": True,
        **cfg["training_model_parameters"],
        **cfg["shared_parameters"],
        "failure_analyzer": analyzer,
    }
    
    mdl = mdl_class.load_from_checkpoint(
        mdl_path,
        **kwargs,
    ).to(device)
    dm.setup("fit")
    mdl.setup("fit")
    dl = dm.val_dataloader()[DatasetType.VAL]
    
    # Limit batches in debug mode
    total_batches = len(dl) if debug_n_batches is None else min(debug_n_batches, len(dl))
    
    try:
        for ii, batch in enumerate(tqdm(dl, total=total_batches)):
            if debug_n_batches is not None and ii >= debug_n_batches:
                break
            batch = {key: val.to(device) for key, val in batch.items()}
            mdl.trajectory_validation_step(batch, DatasetType.VAL)

    finally:    
        print_stats(mdl, analyzer, failure_output_dir)


def print_stats(
        mdl: PretrainingMotionPolicyTransformer | ROPEMotionPolicyTransformer,
        analyzer: FailureAnalyzer | None,
        failure_output_dir: str):
    collision_rate = mdl.val_collision_rate.compute().item()
    reaching_success_rate = mdl.val_reaching_success_rate.compute().item()
    success_rate = mdl.val_success_rate.compute().item()
    position_error = mdl.val_position_error.compute().item()
    orientation_error = mdl.val_orientation_error.compute().item()

    print("Collision Rate:", f"{collision_rate:.2%}")
    print("Reaching Success Rate:", f"{reaching_success_rate:.2%}")
    print("Success Rate:", f"{success_rate:.2%}")
    print("Target Position Error:", f"{100 * position_error:.2}cm",)
    print("Target Orientation Error:", f"{orientation_error:.2}\N{DEGREE SIGN}",)

    if analyzer is not None:
        print(f"Saved {analyzer.num_saved} failed trajectories to: {failure_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to the configuration YAML file")

    # debug mode: only process a limited number of batches
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        metavar="N",
        help="Debug mode: only process N batches (e.g., --debug 10)",
    )

    # failure analysis arguments: enable saving failed trajectories and configure limits
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Enable saving of failed trajectories for analysis",
    )
    parser.add_argument(
        "--failure-output-dir",
        type=str,
        default="datasets/failed_trajectories",
        metavar="PATH",
        help="Directory to save failed trajectory HDF5 files (default: datasets/failed_trajectories)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of collision trajectories to save. If reached with --abort-on-limit, stop validation",
    )
    parser.add_argument(
        "--abort-on-limit",
        action="store_true",
        help="Abort validation when max-failures limit is reached (otherwise just skip saving)",
    )

    # run validation
    args = parser.parse_args()
    run(
        args.cfg_path,
        debug_n_batches=args.debug,
        save_failures=args.save_failures,
        failure_output_dir=args.failure_output_dir,
        max_failures=args.max_failures,
        abort_on_failure_limit=args.abort_on_limit,
    )

"""Specifically runs the validation loop rollout for a jobconfig."""

import argparse

import torch
import yaml
from tqdm.auto import tqdm
from lightning.fabric import Fabric

from avoid_everything_except_exploration.data_loader import DataModule
from avoid_everything_except_exploration.col import CoLMotionPolicyTrainer

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

@torch.no_grad()
def run(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    fabric = Fabric(accelerator="gpu", devices=1, precision="32-true")
    fabric.launch()
    dm = DataModule(
        train_batch_size=cfg["train_batch_size"],
        val_batch_size=cfg["val_batch_size"],
        num_workers=cfg["num_workers"],
        **cfg["data_module_parameters"],
        **cfg["shared_parameters"],
    )
    dm.setup("fit")
    dl = fabric.setup_dataloaders(dm.val_trajectory_dataloader(), move_to_device=True)

    print(f"Loading checkpoint from {cfg['load_checkpoint_path']}")

    trainer = CoLMotionPolicyTrainer.load_from_checkpoint(
        cfg["load_checkpoint_path"],
        **cfg["training_model_parameters"],
        **cfg["shared_parameters"],
        actor_only=True,
    )
    trainer.setup(fabric)

    for batch in tqdm(dl):
        batch = {key: val.cuda() for key, val in batch.items()}
        trainer.trajectory_validation_step(batch)
    print("Collision Rate:", f"{trainer.val_collision_rate.compute().item():.2%}")
    print("Funnel Collision Rate:", f"{trainer.val_funnel_collision_rate.compute().item():.2%}")
    print("Reaching Success Rate:", f"{trainer.val_reaching_success_rate.compute().item():.2%}")
    print("Success Rate:", f"{trainer.val_success_rate.compute().item():.2%}")
    print("Mean Waypoint Count:", f"{trainer.val_waypoint_count.compute().item():.2f}")
    print("Mean Step Size, unnormalized:", f"{trainer.val_step_size_unnorm.compute().item():.3f}")
    print("Mean Step Size, normalized:", f"{trainer.val_step_size_norm.compute().item():.3f}")
    print(
        "Target Position Error:",
        f"{100 * trainer.val_position_error.compute().item():.2}cm",
    )
    print(
        "Target Orientation Error:",
        f"{trainer.val_orientation_error.compute().item():.2}\N{DEGREE SIGN}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path")
    args = parser.parse_args()
    run(args.cfg_path)

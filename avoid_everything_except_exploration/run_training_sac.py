"""
This file contains the run() function, which is responsible for running
the CoL training procedure.
"""

# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
import random
import gc
from pathlib import Path
from typing import Any, Dict
import time
from contextlib import contextmanager

from tqdm import tqdm
from termcolor import cprint
# from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger  # Fabric can use PL loggers
import numpy as np
import torch
import yaml
from collections import defaultdict

from avoid_everything_except_exploration.data_loader import DataModule
from avoid_everything_except_exploration.mixed_batch_provider import MixedBatchProvider
from avoid_everything_except_exploration.replay import ReplayBuffer
from avoid_everything_except_exploration.mixed_batch_provider import AsyncReplay
from avoid_everything_except_exploration.sac import SACMotionPolicyTrainer
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer


torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")

def setup_logger(
    should_log: bool,
    experiment_name: str,
    config_values: Dict[str, Any],
    project_name: str = "avoid-everything-except-exploration",
) -> WandbLogger | None:
    if not should_log:
        return None
    logger = WandbLogger(name=experiment_name, project=project_name, log_model=True)
    logger.log_hyperparams(config_values)
    return logger

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def pretty_k(n):  # e.g., 123_456 -> '123.5k'
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}k" if n >= 1e3 else str(n))

def parse_args_and_configuration():
    """
    Checks the command line arguments and merges them with the configuration yaml file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    with open(args.yaml_config) as f:
        configuration = yaml.safe_load(f)

    return {
        "training_node_name": os.uname().nodename,
        **configuration,
        **vars(args),
    }


def run():
    """
    Runs the CoL training procedure
    """
    config = parse_args_and_configuration()
    logger = setup_logger(config["logging"], config["experiment_name"], config)

    max_train_batches = None
    if config["mintest"]:
        # restrict number of batches for debugging runs
        max_train_batches = 2000

    base_save_dir = config["save_checkpoint_dir"]
    run_id = str(logger.version) if logger is not None else "default"
    save_dir = Path(base_save_dir) / run_id
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # fabric = Fabric(accelerator=device, devices=config["n_gpus"], precision="32-true")
    # fabric = Fabric(accelerator=device, devices=config["n_gpus"], precision="16-mixed")
    # fabric.launch()
    is_rank_zero: bool = True #fabric.global_rank == 0

    if is_rank_zero:
        print("Using device:", device)
        cprint(f"Experiment name: {config['experiment_name']}", "blue")
        if not config["logging"]:
            cprint("Logs disabled", "red")

    # make deterministic-ish
    seed = 42 #+ fabric.global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dm = DataModule(
        train_batch_size=8 if config["mintest"] else config["prefill_batch_size"],
        val_batch_size=8 if config["mintest"] else config["val_batch_size"],
        num_workers=(
            0 if config["mintest"] else config["num_workers"]
        ),
        **(config["shared_parameters"] or {}),
        **(config["data_module_parameters"] or {}),
    )
    dm.setup("fit")
    # train_trajectory_loader = fabric.setup_dataloaders(dm.train_trajectory_dataloader(), move_to_device=True)
    train_trajectory_loader = dm.train_trajectory_dataloader(batch_size=config["prefill_batch_size"]) 
    print(f"Workers for train trajectory loader: {train_trajectory_loader.num_workers}, persistent_workers: {train_trajectory_loader.persistent_workers}, batch_size: {train_trajectory_loader.batch_size}")
    # expert_loader = fabric.setup_dataloaders(dm.train_dataloader(), move_to_device=True)
    # val_state_loader = fabric.setup_dataloaders(
    #     dm.val_state_dataloader(), move_to_device=True)
    # val_trajectory_loader = fabric.setup_dataloaders(
    #     dm.val_trajectory_dataloader(), move_to_device=True)

    replay_buffer = ReplayBuffer(
        capacity=config["replay_buffer_capacity"],
        urdf_path=config["shared_parameters"]["urdf_path"],
        robot_dof=config["training_model_parameters"]["robot_dof"],
        num_robot_points=config["shared_parameters"]["num_robot_points"],
        num_target_points=config["data_module_parameters"]["num_target_points"],
        dataset=dm.data_train,
        pin_memory= device == "cuda", # pin memory if using GPU to speed up transfers
    )
    async_replay = AsyncReplay(
        replay=replay_buffer,
        batch_size=config["train_batch_size"],
        device=device,  # fabric.device,
        prefetch=config["async_replay_prefetch"],
    )
    # mixed_provider = MixedBatchProvider(
    #     expert_loader=expert_loader,
    #     actor_replay=replay_buffer,
    #     # NOTE:
    #     # Async replay uses a background thread that performs CUDA ops.
    #     # In multi-GPU (DDP) runs this can interact badly with NCCL and
    #     # cause collective operation timeouts when ranks desync.
    #     # To avoid CUDA work from non-main threads in distributed runs,
    #     # disable async replay whenever more than one GPU is used.
    #     use_async=(config["n_gpus"] == 1),
    #     async_prefetch=5,
    # )
    if config["load_model_from_checkpoint"]:
        if is_rank_zero:
            cprint(f"Loading model from checkpoint {config['load_checkpoint_path']}", "blue")
        trainer = SACMotionPolicyTrainer.load_from_checkpoint(
            checkpoint_path=config["load_checkpoint_path"],
            map_location=device,
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
            actor_only=config["load_actor_only"],
            device=device,
        )
    else:
        if is_rank_zero:
            cprint("Initializing new model", "blue")
        bc_model = PretrainingMotionPolicyTransformer.load_from_checkpoint(
            checkpoint_path=config["bc_checkpoint_path"],
            map_location=device,
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
            **(config.get("bc_model_parameters") or {}),
        )
        trainer = SACMotionPolicyTrainer(
            **(config["shared_parameters"] or {}),
            **(config["training_model_parameters"] or {}),
            **config["sac_parameters"],
            bc_model=bc_model,
            device=device,
        )

    # clear any cached memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.configure_optimizers()
    trainer.setup() # trainer.setup(fabric)

    if is_rank_zero:
        cprint("Model parameters:", "blue")
        for name, module in {
            "actor": trainer.actor,
            "critic_q1": trainer.qf1,
            "critic_q2": trainer.qf2,
            "target_critic_q1": trainer.qf1_target,
            "target_critic_q2": trainer.qf2_target,
        }.items():
            tot, tr = count_params(module)
            print(f"    {name:14s}  total={pretty_k(tot):>7}  trainable={pretty_k(tr):>7}")

    # --- validation helpers ---
    @contextmanager
    def no_grad_inference():
        with torch.inference_mode():
            yield

    def run_val_epoch_with_bar(
        loader, step_fn, *, desc: str, max_batches: int | None = None,
    ):
        trainer.actor.eval()
        trainer.critic.eval()
        total = len(loader) if max_batches is None else min(max_batches, len(loader))
        val_bar = tqdm(
            total=total,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not is_rank_zero,
        )
        it = 0
        with no_grad_inference():
            for batch in loader:
                step_fn(batch)
                it += 1
                val_bar.update(1)
                if max_batches is not None and it >= max_batches:
                    break
        val_bar.close()
        trainer.actor.train()
        trainer.critic.train()

    def run_state_val_epoch(val_state_loader, max_val_batches=None):
        trainer.reset_state_val_metrics()
        run_val_epoch_with_bar(
            val_state_loader,
            trainer.state_validation_step,
            desc="Val (state)",
            max_batches=max_val_batches,
        )
        return trainer.compute_state_val_metrics()

    def run_rollout_val_epoch(val_trajectory_loader, max_val_batches=None):
        trainer.reset_rollout_val_metrics()
        run_val_epoch_with_bar(
            val_trajectory_loader,
            trainer.trajectory_validation_step,
            desc="Val (rollout)",
            max_batches=max_val_batches,
        )
        return trainer.compute_rollout_val_metrics()
    # --- ---

    train_trajectory_loader_iterator = iter(train_trajectory_loader) # TODO: what if it finishes? should we reset it?
    progress_bar = tqdm(
            total=replay_buffer.capacity,
            desc=f"Prefill buffer",
            unit="transition",
            leave=True,
            disable=not is_rank_zero,
            dynamic_ncols=True,
        )
    while len(replay_buffer) < replay_buffer.capacity/10:
        # idx, q, a, q_next, r, done = trainer.simulation_step(train_trajectory_loader_iterator)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            idx, q, a, q_next, r, done = trainer.simulation_step_v2(train_trajectory_loader_iterator)
        replay_buffer.push(
            idx=idx,
            q=q,
            a=a,
            q_next=q_next,
            r=r,
            done=done.unsqueeze(1),
        )
        progress_bar.update(len(done))
    progress_bar.close()
    if is_rank_zero:
         cprint(f"Initial replay buffer filled with {len(replay_buffer)} transitions.", "green")

    # --- training loop ---
    # train_trajectory_loader = fabric.setup_dataloaders(dm.train_trajectory_dataloader(batch_size=config["simulation_batch_size"]), move_to_device=True) # re-setup dataloader with new batch size
    train_trajectory_loader = dm.train_trajectory_dataloader(batch_size=config["simulation_batch_size"]) 
    print(f"Workers for train trajectory loader: {train_trajectory_loader.num_workers}, persistent_workers: {train_trajectory_loader.persistent_workers}, batch_size: {train_trajectory_loader.batch_size}")
    # val_trajectory_loader = fabric.setup_dataloaders(
    #     dm.val_trajectory_dataloader(), move_to_device=True)
    val_trajectory_loader = dm.val_trajectory_dataloader()

    n_batches = max_train_batches if max_train_batches is not None else len(train_trajectory_loader)
    last_ckpt_time = time.time()
    global_step = 0
    times = {
        "simulation_time": 0.0,
        "replay_sample_time": 0.0,
        "training_time": 0.0,
    }
    
    train_metrics_accum = defaultdict(lambda: torch.tensor(0.0, device=device))
    train_metrics_count = 0 
    
    for epoch in range(config["max_epochs"]):
        epoch_bar = tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1}/{config['max_epochs']}",
            unit="batch",
            leave=True,
            disable=not is_rank_zero,
            dynamic_ncols=True,
        )

        batch_idx = 0

        for _ in range(n_batches):
            # --- Phase management ---
            pretraining: bool = global_step < config["pretraining_steps"]

            # --- Simulation step ---
            t0 = time.time()
            for _ in range(config["sim_steps_per_train_step"]):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # idx, q, a, q_next, r, done = trainer.simulation_step(train_trajectory_loader_iterator)
                    idx, q, a, q_next, r, done = trainer.simulation_step_v2(train_trajectory_loader_iterator) # TODO: is this a problem? simulating a non-constant num of steps between training steps?
                replay_buffer.push(
                    idx=idx,
                    q=q,
                    a=a,
                    q_next=q_next,
                    r=r,
                    done=done.unsqueeze(1),
                )
            t1 = time.time()
            simulation_time = t1 - t0

            # batch, data_loader_iterations = mixed_provider.sample(
            #     8 if config["mintest"] else config["train_batch_size"],
            #     expert_fraction=config["expert_fraction"],
            #     pretraining=pretraining,
            #     device=fabric.device,
            # )

            # --- Training step ---

            # update_targets: bool = global_step % config["actor_delay"] == 0
            # use_actor_loss: bool = update_targets and (global_step > config["start_using_actor_loss"])
            # batch = replay_buffer.sample(config["train_batch_size"], device=device)
            batch = async_replay.get()  # Blocks until next prefetched batch is ready
            t2 = time.time()
            replay_sample_time = t2 - t1
            with torch.cuda.amp.autocast(dtype=torch.float16):
                train_metrics = trainer.train_step(
                    batch,
                    # fabric=fabric,
                    global_step=global_step,
                )
            training_time = time.time() - t2
            times["simulation_time"] += simulation_time
            times["replay_sample_time"] += replay_sample_time
            times["training_time"] += training_time
            if is_rank_zero and global_step>0 and global_step % 100 == 0:
                print(f"Step {global_step:06d}  sim_time={times['simulation_time']/global_step:.4f}s  replay_sample_time={times['replay_sample_time']/global_step:.4f}s  training_time={times['training_time']/global_step:.4f}s")

            # if global_step % config["collect_rollouts_every_n_steps"] == 0:
            #     actor_rollout_metrics = trainer.actor_rollout(batch, replay_buffer) # fill the replay buffer
            #     if logger:
            #         logger.log_metrics(
            #             {f"train/actor_rollouts/{k}": v for k, v in actor_rollout_metrics.items()},
            #             step=global_step)
            #         logger.log_metrics({"train/replay_buffer_size": len(replay_buffer)}, step=global_step)

            for k, v in train_metrics.items():
                if v is not None:
                    train_metrics_accum[k] = train_metrics_accum[k] + v
            
            train_metrics_count += 1
            
            if logger and (global_step % config["log_train_freq"] == 0) and global_step > 0:
                log_dict = {f"train/{k}": (s / train_metrics_count).item() 
                            for k, s in train_metrics_accum.items()}
                logger.log_metrics(log_dict, step=global_step)
                
                train_metrics_accum = defaultdict(lambda: torch.tensor(0.0, device=device))
                train_metrics_count = 0
            # TODO: log time, episode, validation, replay buffer

            # increment with the number of batches consumed from the expert loader
            # NOTE: This makes it APPEAR as if training slows down to `expert_fraction` of pre-training speed
            prev = batch_idx
            # batch_idx = min(n_batches, batch_idx + data_loader_iterations)
            batch_idx += 1
            if is_rank_zero:
                # epoch_bar.set_postfix(
                #     ordered_dict={
                #         "point_match_loss": metrics["point_match_loss"], 
                #         "pretraining": "True" if pretraining else "False",
                #     })
                epoch_bar.update(batch_idx - prev)
            if batch_idx >= n_batches:
                break

            # if logger and (global_step % config["validate_every_n_steps"] == 0) and global_step>0:
            #     if is_rank_zero:
            #         cprint(f"\nValidation at global step {global_step}", "blue")
            #     val_metrics = run_state_val_epoch(
            #         val_state_loader, max_val_batches=config["mid_epoch_max_val_batches"])
            #     val_metrics.update(run_rollout_val_epoch(
            #         val_trajectory_loader, max_val_batches=config["mid_epoch_max_val_rollouts"]))
            #     if logger:
            #         logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

            # # periodic checkpointing based on wall time
            # if config["checkpoint_interval"] > 0 and (time.time() - last_ckpt_time) / 60.0 >= config["checkpoint_interval"]:
            #     ckpt_path = Path(save_dir) / f"fabric-epoch{epoch+1}-step{global_step}.ckpt"
            #     fabric.save(str(ckpt_path), {
            #         "actor": trainer.actor.state_dict(), 
            #         "critic_q1": trainer.qf1.state_dict(), 
            #         "critic_q2": trainer.qf2.state_dict(),
            #         "actor_optimizer": trainer.actor_optimizer.state_dict(), 
            #         "critic_optimizer": trainer.critic_optimizer.state_dict(),
            #         "actor_scheduler": trainer.actor_scheduler.state_dict(),
            #         "critic_scheduler": trainer.critic_scheduler.state_dict(),
            #     },)
            #     cprint(f"Saved checkpoint to {ckpt_path}", "green")
            #     last_ckpt_time = time.time()

            global_step += 1

        # end of epoch validation
        # if is_rank_zero:
        #     cprint(f"\nEnd of epoch {epoch+1} validation", "blue")
        # val_metrics = run_state_val_epoch(
        #     val_state_loader, max_val_batches=config["end_epoch_max_val_batches"])
        # val_metrics.update(run_rollout_val_epoch(
        #     val_trajectory_loader, max_val_batches=config["end_epoch_max_val_rollouts"]))
        # if logger:
        #     logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    async_replay.close()  # cleanly shutdown the async replay background thread
    cprint("Finished training run.", "green")

if __name__ == "__main__":
    run()

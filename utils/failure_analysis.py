"""
Failure trajectory analysis utilities for validation.

Provides FailureAnalyzer class for saving and analyzing failed rolled-out trajectories. 

Output Directory Structure:
===========================
datasets/failed_trajectories/
├── failed_trajectories_YYYYMMDD_HHMMSS.hdf5    # Binary HDF5 file with trajectories and metadata
└── failed_trajectories_YYYYMMDD_HHMMSS.json     # Human-readable metadata summary


HDF5 File Structure (failed_trajectories_YYYYMMDD_HHMMSS.hdf5):
==============================================================

Group Structure:
  /trajectories/               [GROUP]   Container for all failed trajectory groups
  ├── /0/                      [GROUP]   First failed trajectory (global_id=0)
  │   └── trajectory           [DATASET] Trajectory array [T, DOF] (float32, gzip compressed)
  │       @pidx                int       Problem index in original dataset
  │       @position_error      float     Euclidean distance to target position
  │       @orientation_error   float     Angular error to target orientation
  │       @has_reaching_success bool     Whether trajectory reached target
  │       @has_collision       bool     Always True for saved trajectories
  │
  ├── /1/ ...
  │
  └── /N/                      [GROUP]   Nth failed trajectory
      └── trajectory           [DATASET]
          @pidx                int
          @position_error      float
          @orientation_error   float
          @has_reaching_success bool
          @has_collision       bool
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import h5py
import numpy as np
import torch


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, handling device placement."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class FailureAnalyzer:
    """
    Manages saving and analyzing failed trajectories during validation.
    Encapsulates all operations: save, load, and statistics.
    """

    def __init__(
        self,
        output_dir: str = "datasets/failed_trajectories",
        max_failures: Optional[int] = None,
        abort_on_limit: bool = False,
        original_dataset_path: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        load_existing: Optional[str] = None,
        original_dataset = None,
    ):
        """
        Initialize the failure analyzer.

        :param output_dir: Directory to save HDF5 files (auto-timestamped)
        :param max_failures: Max trajectories to save; if limit reached and abort_on_limit=True, raises exception
        :param abort_on_limit: If True, raise exception when max_failures reached; if False, skip saving
        :param original_dataset_path: Path to the original dataset (for metadata storage)
        :param model_checkpoint: Path to the model checkpoint file
        :param model_name: Name/class of the model (e.g., 'PretrainingMotionPolicyTransformer')
        :param load_existing: Path to an existing HDF5 file to load; enabled is set to True if provided
        :param original_dataset: Dataset object for scene reconstruction in load() method
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_failures = max_failures
        self.abort_on_limit = abort_on_limit
        self._original_dataset_path = original_dataset_path
        self._model_checkpoint = model_checkpoint
        self._model_name = model_name
        self._original_dataset = original_dataset
        self._hdf5_path = None
        self._metadata_path = None
        self._creation_timestamp = None
        self.mode = None
        
        # loading mode, read-only
        if load_existing:
            # Load existing file
            self._hdf5_path = Path(load_existing)
            if not self._hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {self._hdf5_path}")
            
            print(f"✓ Loaded existing FailureAnalyzer from: {self._hdf5_path}")
            # Read creation timestamp from file
            with h5py.File(self._hdf5_path, "r") as f:
                self._creation_timestamp = f.attrs.get("creation_timestamp")
            # Infer metadata path from HDF5 path
            self._metadata_path = self._hdf5_path.parent / f"{self._hdf5_path.stem}.json"
            if not self._metadata_path.exists():
                # warning
                print(f"Warning: Metadata file not found: {self._metadata_path}.")    

            self.mode = "r" # only read existing file, no saving
        
        # new file mode, write enabled
        else:
            if self.output_dir is None:
                raise ValueError("output_dir must be provided when not loading an existing file")
            self._creation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._hdf5_path = self.output_dir / f"failed_trajectories_{self._creation_timestamp}.hdf5"
            self._metadata_path = self.output_dir / f"failed_trajectories_{self._creation_timestamp}.json"
            self.mode = "w" # write enabled
            self._create_summary_file() # create initial summary file with metadata

            self.num_saved = 0
            self.num_collisions = 0
            self.cumulative_position_error = 0.0
            self.cumulative_orientation_error = 0.0
            self.cumulative_reaching_success = 0

    def save(
        self,
        batch: dict,
        rollouts: torch.Tensor,
        has_collision: torch.Tensor,
        position_error: torch.Tensor = None,
        orientation_error: torch.Tensor = None,
        has_reaching_success: torch.Tensor = None,
    ) -> tuple:
        """
        Save failed trajectories to HDF5.

        :param batch: Data batch from the dataloader
        :param rollouts: Generated trajectories [B, T, DOF]
        :param has_collision: Boolean tensor [B] indicating which trajectories collided
        :param position_error: Position error tensor [B]
        :param orientation_error: Orientation error tensor [B] 
        :param has_reaching_success: Boolean tensor [B] indicating reaching success
        :return: Tuple of (output_path, num_saved) or (None, 0) if disabled
        :raise RuntimeError: If max_failures limit reached and abort_on_limit=True
        """
        if not has_collision.any():
            return None, 0

        # Ensure output directory exists
        assert self.output_dir is not None, "output_dir must be set for saving"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get indices of failed trajectories
        failed_indices = torch.where(has_collision)[0].cpu().numpy()
        if len(failed_indices) == 0:
            return self._hdf5_path, 0, 0

        # Convert tensors to numpy
        rollouts_np = _tensor_to_numpy(rollouts)
        pidx_np = _tensor_to_numpy(batch["pidx"])
        position_error_np = _tensor_to_numpy(position_error) 
        orientation_error_np = _tensor_to_numpy(orientation_error) 
        has_reaching_success_np = _tensor_to_numpy(has_reaching_success) 
        # Open or create the HDF5 file
        assert self._hdf5_path is not None, "_hdf5_path must be set"
        mode = "a" if self._hdf5_path.exists() else "w"
        num_saved = 0

        with h5py.File(self._hdf5_path, mode) as f:
            # Create or get the group for trajectories
            if "trajectories" not in f:
                f.create_group("trajectories")

            failed_group = f["trajectories"]

            # Find the next available global ID
            existing_ids = [int(k) for k in failed_group.keys()]  # type: ignore
            next_global_id = max(existing_ids) + 1 if existing_ids else 0

            # Save each failed trajectory
            for fail_idx, batch_idx in enumerate(failed_indices):
                global_id = next_global_id + fail_idx

                # Check max_failures limit
                if self.max_failures is not None and global_id >= self.max_failures:
                    if self.abort_on_limit:
                        raise RuntimeError(
                            f"Maximum failures limit ({self.max_failures}) reached. "
                            f"Aborting validation to prevent further saving."
                        )
                    else:
                        break

                traj_group = failed_group.create_group(str(global_id))  # type: ignore

                # Save the trajectory with metadata as attributes
                traj_ds = traj_group.create_dataset(
                    "trajectory", data=rollouts_np[batch_idx], compression="gzip"
                )

                # Store metadata as attributes on the trajectory dataset
                traj_ds.attrs["pidx"] = int(pidx_np[batch_idx]) # problem index in original dataset
                traj_ds.attrs["has_collision"] = True 
                traj_ds.attrs["position_error"] = float(position_error_np[batch_idx])
                traj_ds.attrs["orientation_error"] = float(orientation_error_np[batch_idx])
                traj_ds.attrs["has_reaching_success"] = bool(has_reaching_success_np[batch_idx])

                num_saved += 1
                self.num_collisions += 1
                self.cumulative_position_error += position_error_np[batch_idx]
                self.cumulative_orientation_error += orientation_error_np[batch_idx]
                self.cumulative_reaching_success += has_reaching_success_np[batch_idx]

        self.num_saved += num_saved
        self._update_summary_file()
        return self._hdf5_path, num_saved


    def _create_summary_file(self):
        """ Create json file with summary metadata"""
        if self._metadata_path is None:
            raise RuntimeError("metadata_path must be set for saving metadata")
        
        summary_data = {
            "metadata" : {
                "creation_timestamp": self._creation_timestamp,
                "model_name": self._model_name,
                "model_checkpoint": self._model_checkpoint,
                "original_dataset_path": self._original_dataset_path,
                "hdf5_path": str(self._hdf5_path),
                "metadata_path": str(self._metadata_path),
            },
            "statistics": {}
        }
        with open(self._metadata_path, "w") as f:
            json.dump(summary_data, f, indent=2)


    def _update_summary_file(self):
        """ Update json file with summary statistics
        """

        try:
            # read existing file
            with open(self._metadata_path, "r") as f:
                summary_data = json.load(f)

                # update statistics
                summary_data["statistics"] = {
                    "num_saved": self.num_saved,
                    "num_collisions": self.num_collisions,
                    "collision_rate": self.num_collisions / self.num_saved if self.num_saved > 0 else None,
                    "mean_position_error": self.cumulative_position_error / self.num_saved if self.num_saved > 0 else None,
                    "mean_orientation_error": self.cumulative_orientation_error / self.num_saved if self.num_saved > 0 else None,
                    "reaching_success_rate": self.cumulative_reaching_success / self.num_saved if self.num_saved > 0 else None,
                }

            # overwrite file
            with open(self._metadata_path, "w") as f:
                json.dump(summary_data, f, indent=2)

        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found: {self._metadata_path}.")
        

    def load(self, trajectory_id: int, load_scene: bool = True) -> dict:
        """
        Load a single failed trajectory from the HDF5 file.

        :param trajectory_id: Global ID of the trajectory to load
        :param load_scene: If True, load the full scene from the original dataset
        :return: Dictionary containing trajectory, metadata, and optionally scene data
        """
        if self._hdf5_path is None:
            raise RuntimeError(f"No trajectory dataset exists at {self._hdf5_path}")

        with h5py.File(self._hdf5_path, "r") as f:
            traj_group = f["trajectories"][str(trajectory_id)]  
            traj_ds = traj_group["trajectory"] 
            
            # Extract metadata from trajectory dataset attributes
            metadata = dict(traj_ds.attrs) # pidx, has_collision, position_error, orientation_error, has_reaching_success
            result = {
                "trajectory": traj_ds[:],  # type: ignore
                **metadata, # add all metadata attributes to the result dictionary
            }
        
        # Optionally load scene from original dataset
        if load_scene and self._original_dataset is not None:
            pidx = int(metadata["pidx"])  # get problem index to load scene from original dataset
            scene_data = self._original_dataset[pidx]
            result.update(
                **scene_data # unpack all key-value pairs from scene_data into result
            )
        
        return result
    

    def __repr__(self) -> str:
        status = f"mode={self.mode}, saved={self.num_saved}"
        if self.max_failures:
            status += f", max={self.max_failures}"
        return f"FailureAnalyzer({status})"



class SavedTrajectoryDataset(torch.utils.data.Dataset):
    """
        TrajectoryDataset wrapper to load all saved trajectories and embed them in the original dataset
    """
    def __init__(self, failure_analyzer: "FailureAnalyzer"):
        self.failure_analyzer = failure_analyzer

    def __len__(self):
        return self.failure_analyzer.num_saved

    def __getitem__(self, idx):
        return self.failure_analyzer.load(idx)
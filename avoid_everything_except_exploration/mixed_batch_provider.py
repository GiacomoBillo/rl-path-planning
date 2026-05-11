"""
Mixed batch sampling utilities for Cycle-of-Learning.

This module streams expert transitions from a DataLoader and mixes them with
actor transitions sampled from a replay sampler.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple
import threading
import queue
import traceback
import torch

from avoid_everything_except_exploration.replay import ReplayBuffer


class AsyncReplay:
    """
    A class that prefetches actor samples from a replay buffer asynchronously.
    """

    def __init__(
        self,
        replay: ReplayBuffer,
        batch_size: int,
        device: torch.device | str,
        *,
        prefetch: int = 2,
        warmup_min: int | None = None,
        backoff: float = 0.01,           # seconds between retries
    ):
        self._replay = replay
        self._batch_size = int(batch_size)
        self._device = torch.device(device)
        self._q: queue.Queue[dict[str, torch.Tensor]] = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._warmup_min  = int(warmup_min if warmup_min is not None else prefetch * self._batch_size)
        self._backoff     = float(backoff)
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while not self._stop.is_set():
            if len(self._replay) < self._warmup_min:
                self._stop.wait(self._backoff)
                continue
            try:
                batch = self._replay.sample(self._batch_size, device=self._device)
                self._q.put(batch, timeout=0.1)
                # print(f"[AsyncReplay] put batch: {self._q.qsize()}")
            except queue.Full:
                continue
            except (AssertionError, ValueError) as e:
                # “replay underflow” or a bad row while buffer is filling/rotating.
                # Backoff and retry instead of crashing the thread.
                # print(f"[AsyncReplay] transient sampling error: {e}")
                self._stop.wait(self._backoff)
                continue
            except Exception as e:
                # log but keep the thread alive
                # print(f"AsyncReplay EXCEPTION: {e}")
                traceback.print_exc()
                self._stop.wait(self._backoff)
                continue

    def get(self) -> dict[str, torch.Tensor]:
        """Block until a prefetched batch is ready"""
        return self._q.get()

    def close(self):
        self._stop.set()
        try:
            # drain so the thread is not stuck on put()
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        self._t.join(timeout=1.0)

def _to_tensor(x: Any) -> torch.Tensor:
    """Convert a scalar or tensor-like to a torch.Tensor

    This helper standardizes values so that items returned from the expert
    loader can be collated uniformly.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")


def _collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample dictionaries into a batched dictionary.

    Expects each dictionary in `items` to share the same keys and that each
    value is a 1D or multi-D tensor representing a single sample. Stacking
    occurs along a new batch dimension at dim=0 for every key.

    Returns:
        A dict[str, torch.Tensor] with tensors of shape [B, ...].
    """
    if len(items) == 0:
        return {}
    keys = items[0].keys()
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [_to_tensor(it[k]) for it in items]
        batch[k] = torch.stack(vals, dim=0)
    return batch


class MixedBatchProvider:
    """
    Streams expert transitions from a DataLoader and mixes them with actor samples
    drawn from a replay sampler.

    The provider does not pre-load or store the entire expert dataset; instead it
    maintains a small pool to bridge DataLoader batch boundaries. This
    enables precise batch assembly with minimal memory while staying compatible
    with typical PyTorch DataLoader usage.
    """

    def __init__(
        self,
        expert_loader: Iterable[Any],
        actor_replay: ReplayBuffer,
        use_async: bool = True,
        async_prefetch: int = 3,
        *,
        key_renames: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a mixed-batch provider.

        Parameters:
            expert_loader: An iterable (e.g., a DataLoader) that yields batched
                dictionaries with consistent keys. These represent expert
                transitions and must match the schema expected by training.
            key_renames: Optional mapping to rename incoming keys from the 
                expert loader (for example, {"supervision": "next_configuration"}).
                Applied per-sample when items are pulled from the expert loader.
        """
        self._expert_loader = expert_loader
        self._expert_iter = iter(self._expert_loader)
        self._expert_pool: List[Dict[str, torch.Tensor]] = []
        self._key_renames = key_renames or {}

        self._actor_replay = actor_replay
        self._use_async = use_async
        self._async: Optional[AsyncReplay] = None
        self._async_bs: Optional[int] = None
        self._async_dev: Optional[torch.device] = None
        self._async_prefetch: int = async_prefetch

    def _split_batch_to_items(
        self,
        expert_batch: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Split a batched expert dictionary into a list of per-sample dicts.

        Each tensor value in the input dict is sliced along dim=0 to produce
        one dictionary per sample. Key renames are applied.
        """
        # Determine batch size by first tensor entry
        size = None
        for v in expert_batch.values():
            if isinstance(v, torch.Tensor):
                size = v.shape[0]
                break
        if size is None:
            raise ValueError("Batch must contain at least one tensor with a batch dimension")
        items: List[Dict[str, torch.Tensor]] = []
        for i in range(size):
            item: Dict[str, torch.Tensor] = {}
            for k, v in expert_batch.items():
                vv = v[i] if isinstance(v, torch.Tensor) else _to_tensor(v)
                # Apply key mapping if needed (e.g., supervision -> next_configuration)
                dst_key = self._key_renames.get(k, k)
                item[dst_key] = vv.detach() if isinstance(vv, torch.Tensor) else _to_tensor(vv)
            items.append(item)
        return items

    def _fill_expert_pool(self, required_samples: int) -> int:
        """Fill the internal expert pool up to `required_samples` items.

        Pulls batches from the expert iterator and appends per-sample entries
        to the pool until the requested minimum is satisfied. The iterator is
        reset upon exhaustion (StopIteration), providing an infinite stream if
        desired.

        Returns:
            The number of data loader iterations used to fill the pool.
        """
        data_loader_iterations = 0
        while len(self._expert_pool) < required_samples:
            try:
                batch = next(self._expert_iter)
            except StopIteration:
                self._expert_iter = iter(self._expert_loader)
                batch = next(self._expert_iter)
            data_loader_iterations += 1
            self._expert_pool.extend(self._split_batch_to_items(batch))
        return data_loader_iterations

    def _pop_expert(self, expert_samples: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Remove and return `expert_samples` items from the pool as a batched dict.

        Parameters:
            expert_samples: Number of expert samples to return. If zero, returns an empty dict.

        Returns:
            A dictionary with tensors of shape [expert_samples, ...] and the 
            number of data loader iterations used to fill the pool.
        """
        if expert_samples <= 0:
            return {}, 0
        data_loader_iterations = self._fill_expert_pool(expert_samples)
        items = self._expert_pool[:expert_samples]
        del self._expert_pool[:expert_samples]
        return _collate(items), data_loader_iterations

    def _ensure_async(self, n_actor_samples: int, device: torch.device):
        """(Re)create the async prefetcher if shape/device changed."""
        if n_actor_samples <= 0:
            # pretraining: no actor samples needed; shut down to save CPU
            if self._async is not None:
                self._async.close()
                self._async = None
                self._async_bs = None
                self._async_dev = None
            return
        if (self._async is None or
            self._async_bs != n_actor_samples or
            self._async_dev != device):
            # (Re)start with current actor batch size and device
            if self._async is not None:
                self._async.close()
            self._async = AsyncReplay(
                self._actor_replay,
                n_actor_samples,
                device,
                prefetch=self._async_prefetch,
            )
            self._async_bs = n_actor_samples
            self._async_dev = device

    def sample(
        self,
        total_batch_size: int,
        expert_fraction: float,
        pretraining: bool,
        device: torch.device | str | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Draw a mixed batch of expert and actor transitions.

        During pretraining (``pretraining=True``) the batch is 100% expert. In
        CoL RL-finetuning mode (``pretraining=False``), this returns 
        expert_fraction of the batch from the expert stream and the 
        remainder from the actor replay.

        Parameters:
            total_batch_size: Exact batch size to return.
            expert_fraction: Fraction of samples to sample from expert loader during fine-tuning.
            pretraining: If True, returns only expert samples.
            device: Optional target device for actor samples (falls back to expert batch device).

        Returns:
            A dictionary of tensors representing a full batch and the number of 
            expert data loader iterations used to fill the batch.
        """
        if pretraining:
            try:
                b = next(self._expert_iter)
            except StopIteration:
                self._expert_iter = iter(self._expert_loader)
                b = next(self._expert_iter)
            return b, 1

        assert expert_fraction >= 0.0 and expert_fraction <= 1.0
        n_expert_samples = int(round(total_batch_size * expert_fraction))
        n_actor_samples = total_batch_size - n_expert_samples

        expert_batch, data_loader_iterations = self._pop_expert(n_expert_samples) if n_expert_samples > 0 else ({}, 0)

        # choose a target device based on expert batch (if present)
        if device is None:
            if expert_batch:
                for v in expert_batch.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
        assert isinstance(device, torch.device)

        if self._use_async:
            self._ensure_async(n_actor_samples, device)
            actor_batch: Dict[str, torch.Tensor] = {}
            if n_actor_samples > 0:
                assert self._async is not None, "AsyncR eplay should be initialized for actor samples"
                actor_batch = self._async.get()
        else:
            actor_batch  = self._actor_replay.sample(n_actor_samples, device=device) if n_actor_samples > 0 else {}

        if n_expert_samples == 0:
            return actor_batch, 0
        if n_actor_samples == 0:
            return expert_batch, data_loader_iterations

        common = expert_batch.keys() & actor_batch.keys()
        merged = {k: torch.cat([expert_batch[k], actor_batch[k]], dim=0) for k in common}
        
        any_tensor = next(iter(merged.values()))
        perm = torch.randperm(total_batch_size, device=any_tensor.device)
        for k in merged:
            merged[k] = merged[k][perm]

        return merged, data_loader_iterations

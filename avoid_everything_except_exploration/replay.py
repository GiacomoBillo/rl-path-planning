import threading
import torch

from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler
from avoid_everything_except_exploration.data_loader import Base


class ReplayBuffer:
    """
    Replay buffer for the CoL algorithm.
    Stores minimal transitions (idx, q, a, q_next, r, done) and reconstructs the 
    scene state using the idx when sampling.
    """

    def __init__(
        self,
        capacity: int, urdf_path: str, robot_dof: int,
        num_robot_points: int,
        num_target_points: int,
        dataset: Base,
        pin_memory: bool = True,
    ):
        self.capacity = int(capacity)
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.urdf_path = urdf_path
        self.robot = None
        self.robot_sampler = None
        self.dataset = dataset # reference to the dataset (for extracting scene info)
        self.pin_memory = pin_memory

        # data pinned to CPU memory (RAM)
        self.idx   = torch.empty(capacity, dtype=torch.int64,  pin_memory=pin_memory)
        self.q     = torch.empty(capacity, robot_dof, dtype=torch.float16, pin_memory=pin_memory)
        self.a = torch.empty(capacity, robot_dof, dtype=torch.float16, pin_memory=pin_memory)
        self.qnext = torch.empty(capacity, robot_dof, dtype=torch.float16, pin_memory=pin_memory)
        self.r     = torch.empty(capacity, 1, dtype=torch.float32, pin_memory=pin_memory)
        self.done  = torch.empty(capacity, 1, dtype=torch.uint8,  pin_memory=pin_memory)
        self.ptr = 0
        self.full = False

        self._lock = threading.Lock()
        self.robot_dof = robot_dof
        # Cache pinned batch buffers keyed by batch_size to avoid losing pinned status on indexing
        self._batch_buffers = {}

    def _get_or_create_batch_buffer(self, batch_size: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get or create a pinned batch buffer for the given batch size."""
        key = (batch_size, shape, dtype)
        if key not in self._batch_buffers:
            buffer_shape = (batch_size,) + shape
            self._batch_buffers[key] = torch.empty(
                buffer_shape,
                dtype=dtype,
                pin_memory=self.pin_memory
            )
        return self._batch_buffers[key]

    def _ensure_robot_sampler(self, device: torch.device):
        if self.robot_sampler is None or self.robot is None:
            self.robot = Robot(self.urdf_path, device=device)
            self.robot_sampler = TorchRobotSampler(
                self.robot,
                num_robot_points=self.num_robot_points,
                num_eef_points=self.num_target_points,
                with_base_link=True,
                use_cache=True,
                device=device,
            )
        # assert self.robot_sampler.device == device, "Sampler device mismatch"
        assert self.robot.device == device, "Robot device mismatch"

    def push(self, idx, q, a, q_next, r, done):
        """
        Add a transition to the replay buffer. Expects tensors on GPU or CPU; 
        moves them to CPU pinned asynchronously. Dtype conversion happens during transfer.

        :param idx: [B]
        :param q: [B, DOF], in normalized configuration space ([-1, 1])
        :param a: [B, DOF], in normalized configuration space
        :param q_next: [B, DOF], in normalized configuration space
        :param r: [B, 1]
        :param done: [B, 1]
        """
        B = idx.shape[0]
        with self._lock:
            end = self.ptr + B
            if end <= self.capacity:
                sl = slice(self.ptr, end)
                # copy_() handles dtype conversion and device transfer asynchronously
                self.idx[sl].copy_(idx,      non_blocking=True)
                self.q[sl].copy_(q,          non_blocking=True)
                self.a[sl].copy_(a,          non_blocking=True)
                self.qnext[sl].copy_(q_next, non_blocking=True)
                self.r[sl].copy_(r,          non_blocking=True)
                self.done[sl].copy_(done,    non_blocking=True)
            else:
                first = self.capacity - self.ptr
                sl1 = slice(self.ptr, self.capacity)
                sl2 = slice(0, end - self.capacity)

                self.idx[sl1].copy_(idx[:first],      non_blocking=True)
                self.q[sl1].copy_(q[:first],          non_blocking=True)
                self.a[sl1].copy_(a[:first],          non_blocking=True)
                self.qnext[sl1].copy_(q_next[:first], non_blocking=True)
                self.r[sl1].copy_(r[:first],          non_blocking=True)
                self.done[sl1].copy_(done[:first],    non_blocking=True)

                self.idx[sl2].copy_(idx[first:],      non_blocking=True)
                self.q[sl2].copy_(q[first:],          non_blocking=True)
                self.a[sl2].copy_(a[first:],          non_blocking=True)
                self.qnext[sl2].copy_(q_next[first:], non_blocking=True)
                self.r[sl2].copy_(r[first:],          non_blocking=True)
                self.done[sl2].copy_(done[first:],    non_blocking=True)

            self.ptr  = end % self.capacity
            # Mark as full once we've written at least `capacity` entries total,
            # even if the write did not land exactly on ptr == 0.
            self.full = self.full or (end >= self.capacity)

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        """
        device = torch.device(device) if device is not None else torch.device('cpu')

        with self._lock:
            n = len(self)
            assert n >= batch_size, "replay underflow"
            ids = torch.randint(n, (batch_size,), device='cpu')

            # Get or create pinned batch buffers for this batch size
            idx_buf   = self._get_or_create_batch_buffer(batch_size, (), torch.int64)
            q_buf     = self._get_or_create_batch_buffer(batch_size, (self.robot_dof,), torch.float32)
            a_buf     = self._get_or_create_batch_buffer(batch_size, (self.robot_dof,), torch.float32)
            qn_buf    = self._get_or_create_batch_buffer(batch_size, (self.robot_dof,), torch.float32)
            r_buf     = self._get_or_create_batch_buffer(batch_size, (1,), torch.float32)
            done_buf  = self._get_or_create_batch_buffer(batch_size, (1,), torch.float32)
            
            # Copy indexed data into pinned buffers (CPU-to-CPU, preserves pinned memory status)
            # otherwise indexing returns non-pinned tensors, which would cause GPU transfers to be synchronous and slow
            idx_buf.copy_(self.idx[ids])
            q_buf.copy_(self.q[ids].to(dtype=torch.float32))
            a_buf.copy_(self.a[ids].to(dtype=torch.float32))
            qn_buf.copy_(self.qnext[ids].to(dtype=torch.float32))
            r_buf.copy_(self.r[ids])
            done_buf.copy_(self.done[ids].to(dtype=torch.float32))
            
            # Keep references to buffers while lock is held
            idx   = idx_buf
            q     = q_buf
            a     = a_buf
            qn    = qn_buf
            r     = r_buf
            done  = done_buf

        max_idx = int(len(self.dataset))
        bad = (idx < 0) | (idx >= max_idx)
        if bad.any():
            # Let AsyncReplay retry instead of crashing the thread:
            raise ValueError(f"ReplayBuffer invalid indices: {int(bad.sum())}/{bad.numel()}")

        # (after releasing lock) move pinned tensors to device asynchronously
        idx  = idx.to(device, non_blocking=True)
        q    = q.to(device, non_blocking=True)
        a    = a.to(device, non_blocking=True)
        qn   = qn.to(device, non_blocking=True)
        r    = r.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)

        # only load unique scenes in the batch, then transfer to device and expand back to batch size using inv indices
        uniq, inv = torch.unique(idx, sorted=True, return_inverse=True)
        sc = self.dataset.batch_scenes_by_idx(uniq.cpu(), pin=self.pin_memory)  # CPU pinned
        
        def to_dev_fast(x):  # x: CPU pinned [U, ...], inv: GPU tensor
            # Transfer pinned tensor to GPU (async), then expand on GPU to avoid roundtrips
            x_dev = x.to(device, non_blocking=True)  # [U, ...] -> GPU
            return x_dev[inv]  # [U, ...] -> [B, ...] on GPU
        
        cuboid_centers    = to_dev_fast(sc["cuboid_centers"])
        cuboid_dims       = to_dev_fast(sc["cuboid_dims"])
        cuboid_quats      = to_dev_fast(sc["cuboid_quats"])
        cylinder_centers  = to_dev_fast(sc["cylinder_centers"])
        cylinder_radii    = to_dev_fast(sc["cylinder_radii"])
        cylinder_heights  = to_dev_fast(sc["cylinder_heights"])
        cylinder_quats    = to_dev_fast(sc["cylinder_quats"])
        target_position   = to_dev_fast(sc["target_position"])
        target_orientation= to_dev_fast(sc["target_orientation"])
        scene_points      = to_dev_fast(sc["scene_points"])
        target_points     = to_dev_fast(sc["target_points"])


        self._ensure_robot_sampler(device)
        assert self.robot is not None
        assert self.robot_sampler is not None

        # state reconstruction: unnormalize q, sample robot point cloud, update point cloud and labels from dataset
        q_unn = self.robot.unnormalize_joints(q)
        robot_points = self.robot_sampler.sample(q_unn)[..., :3]  # [B, N_robot, 3]
        pc = torch.cat([robot_points, scene_points, target_points], dim=1)  # [B, N_total, 3]
        B = pc.size(0)
        if not hasattr(self, "_labels"):
            R, S, T = self.num_robot_points, self.dataset.num_obstacle_points, self.num_target_points
            self._labels = torch.cat([torch.zeros(R,1), torch.ones(S,1), 2*torch.ones(T,1)], dim=0)
            if self.pin_memory:
                self._labels = self._labels.pin_memory()
        labels = self._labels.expand(B, -1, -1).to(device, non_blocking=True)

        # next state reconstruction
        qn_unn = self.robot.unnormalize_joints(qn)
        robot_points_next = self.robot_sampler.sample(qn_unn)[..., :3]  # [B, N_robot, 3]
        pc_next = torch.cat([robot_points_next, scene_points, target_points], dim=1)  # [B, N_total, 3]

        batch = {
            "idx": idx,
            # "configuration": q,
            "state": {
                "configuration": q,
                "point_cloud": pc,
                "point_cloud_labels": labels,
            },
            "action": a,
            # "next_configuration": qn,
            "next_state": {
                "configuration": qn,
                "point_cloud": pc_next,
                "point_cloud_labels": labels, # labels next = labels (fixed number of points per category)
            },
            "reward": r,
            "done": done,
            # "point_cloud": pc,
            # "point_cloud_labels": labels,
            "cuboid_centers": cuboid_centers,
            "cuboid_dims": cuboid_dims,
            "cuboid_quats": cuboid_quats,
            "cylinder_centers": cylinder_centers,
            "cylinder_radii": cylinder_radii,
            "cylinder_heights": cylinder_heights,
            "cylinder_quats": cylinder_quats,
            "target_position": target_position,
            "target_orientation": target_orientation,
            "is_expert": torch.zeros(B, 1, dtype=torch.float32, device=device),
        }
        return batch

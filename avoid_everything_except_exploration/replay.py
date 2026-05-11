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
    ):
        self.capacity = int(capacity)
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.urdf_path = urdf_path
        self.robot = None
        self.robot_sampler = None
        self.dataset = dataset # reference to the dataset (for extracting scene info)

        # data pinned to CPU memory (RAM)
        self.idx   = torch.empty(capacity, dtype=torch.int64,  pin_memory=True)
        self.q     = torch.empty(capacity, robot_dof, dtype=torch.float16, pin_memory=True)
        self.a     = torch.empty_like(self.q)
        self.qnext = torch.empty_like(self.q)
        self.r     = torch.empty(capacity, 1, dtype=torch.float32, pin_memory=True)
        self.done  = torch.empty(capacity, 1, dtype=torch.uint8,  pin_memory=True)
        self.ptr = 0
        self.full = False

        self._lock = threading.Lock()

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
        assert self.robot_sampler.device == device, "Sampler device mismatch"
        assert self.robot.device == device, "Robot device mismatch"

    def push(self, idx, q, a, q_next, r, done):
        """
        Add a transition to the replay buffer. Expects tensors on GPU or CPU; 
        moves them to CPU pinned asynchronously

        :param idx: [B]
        :param q: [B, DOF], in normalized configuration space ([-1, 1])
        :param a: [B, DOF], in normalized configuration space
        :param q_next: [B, DOF], in normalized configuration space
        :param r: [B, 1]
        :param done: [B, 1]
        """
        idx    = idx.to(dtype=torch.int64)
        q      = q.to(dtype=torch.float16)
        a      = a.to(dtype=torch.float16)
        q_next = q_next.to(dtype=torch.float16)
        r      = r.to(dtype=torch.float32)
        done   = done.to(dtype=torch.uint8)

        B = idx.shape[0]
        with self._lock:
            end = self.ptr + B
            if end <= self.capacity:
                sl = slice(self.ptr, end)
                self.idx[sl].copy_(idx,      non_blocking=True)  # src may be CUDA -> pinned CPU (async)
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

            # snapshot rows on CPU to detach from concurrent writers
            idx   = self.idx[ids].clone()
            q     = self.q[ids].clone().to(dtype=torch.float32)
            a     = self.a[ids].clone().to(dtype=torch.float32)
            qn    = self.qnext[ids].clone().to(dtype=torch.float32)
            r     = self.r[ids].clone()
            done  = self.done[ids].clone().to(dtype=torch.float32)

        max_idx = int(len(self.dataset))
        bad = (idx < 0) | (idx >= max_idx)
        if bad.any():
            # Let AsyncReplay retry instead of crashing the thread:
            raise ValueError(f"ReplayBuffer invalid indices: {int(bad.sum())}/{bad.numel()}")

        # (after releasing lock) move to device
        idx  = idx.to(device, non_blocking=True)
        q    = q.to(device, non_blocking=True)
        a    = a.to(device, non_blocking=True)
        qn   = qn.to(device, non_blocking=True)
        r    = r.to(device, non_blocking=True)
        done = done.to(device, non_blocking=True)

        uniq, inv = torch.unique(idx, sorted=True, return_inverse=True)
        sc = self.dataset.batch_scenes_by_idx(uniq.cpu())  # CPU pinned
        inv_cpu = inv.cpu()
        
        def to_dev(x):  # x: CPU pinned [U, ...]
            return x[inv_cpu].to(device, non_blocking=True)  # -> [B, ...]
        cuboid_centers    = to_dev(sc["cuboid_centers"])
        cuboid_dims       = to_dev(sc["cuboid_dims"])
        cuboid_quats      = to_dev(sc["cuboid_quats"])
        cylinder_centers  = to_dev(sc["cylinder_centers"])
        cylinder_radii    = to_dev(sc["cylinder_radii"])
        cylinder_heights  = to_dev(sc["cylinder_heights"])
        cylinder_quats    = to_dev(sc["cylinder_quats"])
        target_position   = to_dev(sc["target_position"])
        target_orientation= to_dev(sc["target_orientation"])
        scene_points      = to_dev(sc["scene_points"])
        target_points     = to_dev(sc["target_points"])


        self._ensure_robot_sampler(device)
        assert self.robot is not None
        assert self.robot_sampler is not None
        q_unn = self.robot.unnormalize_joints(q)
        robot_points = self.robot_sampler.sample(q_unn)[..., :3]  # [B, N_robot, 3]

        
        pc = torch.cat([robot_points, scene_points, target_points], dim=1)  # [B, N_total, 3]
        B = pc.size(0)
        if not hasattr(self, "_labels"):
            R, S, T = self.num_robot_points, self.dataset.num_obstacle_points, self.num_target_points
            self._labels = torch.cat([torch.zeros(R,1), torch.ones(S,1), 2*torch.ones(T,1)], dim=0).pin_memory()
        labels = self._labels.expand(B, -1, -1).to(device, non_blocking=True)

        batch = {
            "idx": idx,
            "configuration": q,
            "action": a,
            "next_configuration": qn,
            "reward": r,
            "done": done,
            "point_cloud": pc,
            "point_cloud_labels": labels,
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

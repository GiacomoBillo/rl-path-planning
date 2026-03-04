# Collision Checking in `AvoidEverythingEnv`

## How It Works: The Core Idea

All three collision checking modes share the same underlying principle:

> Represent the robot as a set of **collision spheres** attached to each link.  
> For each sphere, evaluate the obstacle's **SDF** (Signed Distance Function) at the sphere's center.  
> If `sdf(sphere_center) < sphere_radius + buffer` → the sphere overlaps the obstacle → **collision**.

Negative SDF = point is inside the obstacle surface.  
SDF < sphere_radius = the sphere overlaps the obstacle surface.

---

## The Three Steps (Same for All Modes)

### Step 1 — FK: place sphere centers in world space
Run forward kinematics to transform each collision sphere from link-local frame to world frame:
- `"franka"` → `nfk.franka_arm_link_fk(q, prismatic)` — hard-coded Franka FK, fastest
- `"spheres"` → `fk_torch(config)` — generic URDF-based FK
- `"torch"` → `torch_urdf.link_fk_batch(config)` — same URDF-based, batched

### Step 2 — SDF: evaluate obstacle distance at each sphere center

| Mode | SDF function | Implementation |
|------|-------------|----------------|
| `"franka"` | `CuboidArray.scene_sdf(centers)` | Numba-jit, vectorized over all cuboids at once |
| `"spheres"` | `cuboid.sdf(centers)` per primitive | Python loop, per-object Numba-jit |
| `"torch"` | `TorchCuboids.sdf(centers)` | PyTorch batched tensor ops |

All three compute the same geometric quantity — the signed distance from each sphere center to the nearest obstacle surface — just with different backends.

### Step 3 — Collision test
```python
collision = any(sdf(sphere_center) < sphere_radius + scene_buffer)
```

---

## Modes and Pros/Cons

### `"franka"` — `FrankaCollisionSpheres` (default)
**File**: `robofin/robofin/old/collision.py`  
**Used by**: `avoid_everything/environments/base.py`, `cubby_environment.py`, `tabletop.py`  
**Primitives cached as**: `[CuboidArray, CylinderArray]`  

| ✅ Pros | ❌ Cons |
|---------|---------|
| Fastest: Numba-jit `scene_sdf` vectorizes over all obstacles | `old/` module — Franka-specific (hard-coded sphere constants) |
| Includes **self-collision** check | Requires `prismatic_joint` argument (gripper opening, fixed at `0.04`) |
| Exact match to expert planner and environment generation code | Not portable to other robots |
| Pure NumPy — no GPU, no sync overhead in CPU Gym env | |

---

### `"spheres"` — `CollisionSpheres` (generic URDF-based)
**File**: `robofin/robofin/collision.py`  
**Primitives cached as**: `[Cuboid, Cylinder, ...]`  

| ✅ Pros | ❌ Cons |
|---------|---------|
| Generic — works with any robot loaded from URDF | Per-primitive Python loop (no obstacle-level batching) |
| Includes **self-collision** check | FK uses `fk_torch` — heavier than hard-coded Franka FK |
| Pure NumPy — CPU-friendly | Newer, less battle-tested than `FrankaCollisionSpheres` |

---

### `"torch"` — `robot.compute_spheres()` + `TorchCuboids/TorchCylinders`
**Files**: `robofin/robofin/robots.py` + `avoid_everything/geometry.py`  
**Used by**: `avoid_everything/pretraining.py:collision_error()`, `avoid_everything/rope.py`  
**Primitives cached as**: `{"cuboids": TorchCuboids, "cylinders": TorchCylinders}`  

| ✅ Pros | ❌ Cons |
|---------|---------|
| **Identical** to training/validation collision metric — consistent reward signal | **No self-collision** check |
| Vectorized SDF over all obstacles simultaneously | PyTorch overhead for single CPU step (host↔device sync) |
| GPU-capable (for future batch env parallelization) | Heavier FK via `torch_urdf.link_fk_batch` |

---

## Why Not GPU in a Gym Env?

Gymnasium environments are CPU-synchronous: every `step()` is a blocking call from the RL training loop. Using GPU tensors inside the env creates:
- **CUDA sync points** that stall the CPU↔GPU pipeline
- **Host↔device transfer overhead** for a single config at a time
- **Complications** with multi-env vectorization (e.g., `SubprocVecEnv` spawns worker processes that don't share CUDA contexts)

For a CPU Gym env, pure NumPy (`"franka"` or `"spheres"`) is the right default. `"torch"` is worth using only if you later vectorize the environment on GPU or need the reward signal to match training exactly.

--- 

#### 1. The Traditional Setup: CPU Env + GPU Agent

Standard `gymnasium` environments (like Classic Control, Box2D, PyBullet, and standard MuJoCo) **do run entirely on the CPU**. They calculate physics step-by-step and return standard Python floats or NumPy arrays.

However, your *training* is usually split:

1. **The Rollout (CPU):** Your agent interacts with the Gym environment to collect data (states, actions, rewards). This happens on the CPU.
2. **The Transfer:** Once you have a batch of data, your RL algorithm (like Stable Baselines3 or CleanRL) converts those NumPy arrays into PyTorch/TensorFlow tensors and sends them to the GPU.
3. **The Update (GPU):** The heavy math—calculating gradients and updating the neural network weights—happens on the GPU.

**The Problem:** Passing data back and forth between the CPU (for the environment) and the GPU (for the neural network) thousands of times per second creates a massive communication bottleneck. Your GPU often sits idle waiting for the CPU physics engine to finish computing the next frame.

---

#### 2. The Modern Solution: GPU-Native Simulators

To fix this bottleneck, the robotics and AI industry created **Hardware-Accelerated Simulators**. These environments are written natively in PyTorch, JAX, or CUDA.

They do not return NumPy arrays; they return GPU tensors directly. This means the physics simulation, the neural network, and the data all live on the GPU. Zero transfer time.

If you want to run your environment on the GPU, you need to use one of these:

* **Brax (by Google):** A physics engine written in JAX. It mimics classic MuJoCo environments but runs blazingly fast on GPUs and TPUs.
* **Isaac Gym / Isaac Sim (by NVIDIA):** The industry standard for complex robotics. It can simulate thousands of high-fidelity robots in parallel entirely on the GPU.
* **MuJoCo MJX (by DeepMind):** A recent addition that ports standard MuJoCo to JAX, allowing standard MuJoCo models to run natively on the GPU.
* **cuRobo:** Highly optimized for robot kinematics and collision checking on the GPU.

---

#### 3. How to maximize your current setup

If you are stuck using a standard CPU Gymnasium environment (because you are using a custom URDF or a specific library), you can still speed things up. You use **Vectorized Environments** (like `gymnasium.vector.AsyncVectorEnv`).

This spins up 16, 32, or 64 copies of your CPU environment on different CPU cores, gathers their data simultaneously, and sends one giant batch to the GPU to keep it fed.



---

## Usage

```python
# Default (franka — fastest, self-collision, matches expert planner) uses old/original robofin library
env = AvoidEverythingEnv(dataloader=dataloader)

# Generic URDF-based (self-collision, portable) uses new/custom/generalized robofin library
env = AvoidEverythingEnv(dataloader=dataloader, collision_mode="spheres")

# Consistent with training validation metric (no self-collision)
env = AvoidEverythingEnv(dataloader=dataloader, collision_mode="torch")

# Add a safety margin of 1cm
env = AvoidEverythingEnv(dataloader=dataloader, scene_buffer=0.01)
```

"""
Learning rate schedule builders for RL training.

Provides utilities to create learning rate schedules from configuration dictionaries.
Uses Stable Baselines3's built-in schedule functions (get_linear_fn, get_schedule_fn).

Includes phase-aware linear schedule that tracks progress per-phase rather than globally,
allowing each training phase (warmup, finetuning) to have independent LR progression.
"""

from typing import Union, Callable, Dict, Any
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn


class PhaseAwareLinearSchedule:
    """
    Linear learning rate schedule that tracks phase-local progress.
    
    Unlike SB3's get_linear_fn which uses global progress across all training,
    this schedule computes progress relative to a specific phase (warmup or finetuning).
    
    This allows each phase to have independent LR progression from start_lr to end_lr,
    even when phases are run sequentially in the same training session.
    
    Args:
        start_lr: Learning rate at the start of this phase
        end_lr: Learning rate at the end of this phase
        phase_start_step: Global step count when this phase begins
        phase_total_steps: Total number of steps in this phase
        total_training_steps: Total steps across all phases (needed to convert progress_remaining to step count)
    
    Example:
        >>> # Warmup phase: steps 0-1000 of 6000 total
        >>> warmup_schedule = PhaseAwareLinearSchedule(
        ...     start_lr=3e-4, end_lr=1e-5,
        ...     phase_start_step=0, phase_total_steps=1000, total_training_steps=6000
        ... )
        >>> # At warmup start (global step 0): progress_remaining=1.0
        >>> warmup_schedule(1.0)  # Returns 3e-4 (start_lr)
        >>> # At warmup end (global step 1000): progress_remaining=0.833
        >>> warmup_schedule(0.833)  # Returns 1e-5 (end_lr)
        >>>
        >>> # Finetuning phase: steps 1000-6000
        >>> finetune_schedule = PhaseAwareLinearSchedule(
        ...     start_lr=1e-4, end_lr=1e-5,
        ...     phase_start_step=1000, phase_total_steps=5000, total_training_steps=6000
        ... )
        >>> # At finetuning start (global step 1000): progress_remaining=0.833
        >>> finetune_schedule(0.833)  # Returns 1e-4 (start_lr) - phase progress is 1.0!
    """
    def __init__(
        self,
        start_lr: float,
        end_lr: float,
        phase_start_step: int,
        phase_total_steps: int,
        total_training_steps: int
    ):
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.phase_start_step = int(phase_start_step)
        self.phase_total_steps = int(phase_total_steps)
        self.total_training_steps = int(total_training_steps)
        
        if self.phase_total_steps <= 0:
            raise ValueError(f"phase_total_steps must be positive, got {phase_total_steps}")
        if self.total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be positive, got {total_training_steps}")
    
    def __call__(self, progress_remaining: float) -> float:
        """
        Compute learning rate based on phase-local progress.
        
        Args:
            progress_remaining: Global progress remaining (1.0 at start, 0.0 at end)
                               Computed by SB3 as: 1 - (current_step / total_training_steps)
        
        Returns:
            Learning rate interpolated based on phase-local progress
        """
        # Convert global progress_remaining to current global step
        # progress_remaining = 1 - (current_step / total_steps)
        # => current_step = (1 - progress_remaining) * total_steps
        current_global_step = (1.0 - progress_remaining) * self.total_training_steps
        
        # Compute phase-local step (relative to phase start)
        phase_local_step = current_global_step - self.phase_start_step
        
        # Clamp to phase bounds
        phase_local_step = max(0.0, min(phase_local_step, self.phase_total_steps))
        
        # Compute phase-local progress (1.0 at phase start, 0.0 at phase end)
        phase_progress_remaining = 1.0 - (phase_local_step / self.phase_total_steps)
        
        # Linear interpolation from start_lr to end_lr based on phase progress
        lr = self.end_lr + (self.start_lr - self.end_lr) * phase_progress_remaining
        
        return lr


def build_lr_schedule(
    config: Union[Dict[str, Any], float],
    phase_start_step: int = 0,
    phase_total_steps: int = None,
    total_training_steps: int = None,
) -> Callable[[float], float]:
    """
    Build a learning rate schedule from configuration.
    
    Can create either global schedules (using SB3's built-in functions) or
    phase-aware linear schedules that track progress independently per training phase.
    
    Args:
        config: Either a float (for constant LR) or a dict with:
            - type: "constant" or "linear"
            - start_lr: Starting learning rate (required)
            - end_lr: Ending learning rate (required for "linear")
        phase_start_step: Global step when this phase begins (for phase-aware schedules)
        phase_total_steps: Total steps in this phase (for phase-aware schedules)
        total_training_steps: Total steps across all phases (for phase-aware schedules)
    
    Returns:
        A callable schedule function: progress_remaining -> learning_rate
        
    Note:
        - For constant schedules, always uses SB3's get_schedule_fn (phase parameters ignored)
        - For linear schedules, uses PhaseAwareLinearSchedule if phase parameters provided,
          otherwise uses SB3's get_linear_fn
    
    Examples:
        # Constant learning rate (phase-independent)
        >>> schedule = build_lr_schedule(3e-4)
        >>> schedule(0.5)
        0.0003
        
        # Phase-aware linear decay for warmup (steps 0-1000 of 6000 total)
        >>> schedule = build_lr_schedule(
        ...     {"type": "linear", "start_lr": 3e-4, "end_lr": 1e-5},
        ...     phase_start_step=0,
        ...     phase_total_steps=1000,
        ...     total_training_steps=6000
        ... )
        >>> schedule(1.0)  # At global step 0
        3e-4
        
        # Phase-aware linear decay for finetuning (steps 1000-6000)
        >>> schedule = build_lr_schedule(
        ...     {"type": "linear", "start_lr": 1e-4, "end_lr": 1e-5},
        ...     phase_start_step=1000,
        ...     phase_total_steps=5000,
        ...     total_training_steps=6000
        ... )
        >>> schedule(0.833)  # At global step 1000, returns 1e-4 (phase start_lr)
        1e-4
    """
    # Determine if we should use phase-aware schedules (only for linear)
    use_phase_aware = (
        phase_total_steps is not None and 
        total_training_steps is not None
    )
    
    # Handle simple float input (constant LR)
    if isinstance(config, (float, int)):
        return get_schedule_fn(float(config))
    
    # Validate config dict
    if not isinstance(config, dict):
        raise ValueError(f"LR config must be a dict or float, got {type(config)}")
    
    schedule_type = config.get("type", "constant")
    start_lr = config.get("start_lr")
    
    if start_lr is None:
        raise ValueError("LR config must specify 'start_lr'")
    
    if schedule_type == "constant":
        # Constant schedules don't need phase awareness
        return get_schedule_fn(float(start_lr))
    
    elif schedule_type == "linear":
        end_lr = config.get("end_lr")
        if end_lr is None:
            raise ValueError("Linear LR schedule requires 'end_lr'")
        
        if use_phase_aware:
            # Use phase-aware linear schedule for per-phase progression
            return PhaseAwareLinearSchedule(
                start_lr=float(start_lr),
                end_lr=float(end_lr),
                phase_start_step=phase_start_step,
                phase_total_steps=phase_total_steps,
                total_training_steps=total_training_steps
            )
        else:
            # Use SB3's global linear schedule
            end_fraction = config.get("end_fraction", 1.0)
            return get_linear_fn(
                start=float(start_lr),
                end=float(end_lr),
                end_fraction=float(end_fraction)
            )
    
    else:
        raise ValueError(
            f"Unknown LR schedule type '{schedule_type}'. "
            f"Supported types: 'constant', 'linear'"
        )





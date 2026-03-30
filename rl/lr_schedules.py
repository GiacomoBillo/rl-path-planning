"""
Learning rate schedule builders for RL training.

Provides utilities to create learning rate schedules from configuration dictionaries.
Uses Stable Baselines3's built-in schedule functions (get_linear_fn, get_schedule_fn).

Includes phase-aware linear schedule that tracks progress per-phase rather than globally,
allowing each training phase (warmup, finetuning) to have independent LR progression.
"""

from typing import Union, Callable, Dict, Any
from stable_baselines3.common.utils import get_schedule_fn


class PhaseAwareLinearSchedule:
    """
    Learning rate schedule that progresses linearly within a specific training phase.
    
    Unlike standard linear schedules that use global training progress, this schedule
    tracks progress relative to a specific phase (e.g., warmup or finetuning). This
    ensures that when switching phases, the LR starts from start_lr regardless of
    how many steps have already been completed.
    
    Args:
        start_lr: Learning rate at the beginning of the phase
        end_lr: Learning rate at the end of the phase
        phase_start_step: Global step number when this phase begins
        phase_total_steps: Total number of steps in this phase
    """
    def __init__(
        self,
        start_lr: float,
        end_lr: float,
        phase_start_step: int,
        phase_total_steps: int,
    ):
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.phase_start_step = int(phase_start_step)
        self.phase_total_steps = int(phase_total_steps)
        
        if self.phase_total_steps <= 0:
            raise ValueError(f"phase_total_steps must be positive, got {phase_total_steps}")
    
    def __call__(self, current_step: int) -> float:
        """
        Compute the learning rate for the current training step.
        
        Args:
            current_step: Current global training step
        
        Returns:
            Learning rate for the current step within this phase
        """
        # Calculate step within this phase
        phase_step = current_step - self.phase_start_step
        
        # Clamp to phase boundaries (handle edge cases at phase transitions)
        phase_step = max(0, min(phase_step, self.phase_total_steps))
        
        # Calculate progress within this phase (0.0 to 1.0)
        phase_progress = phase_step / self.phase_total_steps
        
        # Convert to progress_remaining format (1.0 to 0.0)
        phase_progress_remaining = 1.0 - phase_progress
        
        # Linear interpolation from start_lr to end_lr
        lr = self.end_lr + (self.start_lr - self.end_lr) * phase_progress_remaining
        
        return lr


def build_lr_schedule(
    config: Dict[str, Any],
    phase_start_step: int = 0,
    phase_total_steps: int = None,
) -> Union[float, Callable[[int], float]]:
    """
    Build a learning rate schedule from configuration.
    
    Supports two types of schedules:
    - constant: Fixed learning rate throughout training
    - linear: Linear decay from start_lr to end_lr over the phase
    
    Args:
        config: Dictionary with schedule configuration
            For constant schedules:
                {
                    'type': 'constant',
                    'start_lr': 3e-4
                }
            For linear schedules:
                {
                    'type': 'linear',
                    'start_lr': 1e-4,
                    'end_lr': 1e-5
                }
        phase_start_step: Global step number when this phase begins (required for linear)
        phase_total_steps: Total steps in this phase (required for linear)
    
    Returns:
        Either a float (constant LR) or a callable schedule function
    
    Raises:
        KeyError: If required configuration keys are missing
        ValueError: If schedule type is invalid or required phase parameters are missing
    """
    schedule_type = config['type']
    
    if schedule_type == 'constant':
        # Constant LR - just return the float value
        return config['start_lr']
    
    elif schedule_type == 'linear':
        # Phase-aware linear decay
        start_lr = config['start_lr']
        end_lr = config['end_lr']
        
        # Validate required parameters for linear schedules
        if phase_total_steps is None:
            raise ValueError("phase_total_steps is required for linear schedules")
        
        return PhaseAwareLinearSchedule(
            start_lr=start_lr,
            end_lr=end_lr,
            phase_start_step=phase_start_step,
            phase_total_steps=phase_total_steps,
        )
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")





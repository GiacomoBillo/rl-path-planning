import argparse
from torch.utils.data import DataLoader
from avoid_everything.data_loader import StateDataset
from avoid_everything.type_defs import DatasetType
from robofin.robots import Robot
from rl.environment import AvoidEverythingEnv


RENDER_MODE = "human"

class RandomPolicy:
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        if seed is not None:
            import random, numpy as _np
            random.seed(seed)
            _np.random.seed(seed)
            try:
                self.action_space.seed(seed)
            except Exception:
                pass

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.action_space.sample()


def run_episode(env, policy, render=False, verbose=False, idx_ep=None):
    if idx_ep is not None and verbose:
        print(f"Starting episode {idx_ep}...")
    total_reward = 0.0
    obs, info = env.reset()
    if render:
        env.render()
    steps = 0
    while True:
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            # print(f"Step {steps}: action={action:.1f}, reward={reward:.1f}, terminated={terminated}, truncated={truncated}, info={info}")
            print(f"Step {steps}:\n",
                  f"\taction={[round(a,2) for a in action]},\n",
                  f"\treward={reward:.1f},\n",
                  f"\tterminated={terminated},\n",
                  f"\ttruncated={truncated},\n",
                  f"\tinfo={ {k: round(v, 2) if isinstance(v, float) else v for k, v in info.items()} }")
        total_reward += reward
        steps += 1
        if render:
            env.render()
        if terminated or truncated:
            break
    if idx_ep is not None and verbose:
        print(f"Episode {idx_ep} finished: total_reward={total_reward:.3f}, steps={steps}")
    return {"total_reward": total_reward, "steps": steps}


def evaluate_policy(policy, env, num_episodes=10, render=False, verbose=False):
    metrics = [run_episode(env, policy, render=render, verbose=verbose, idx_ep=idx_ep) for idx_ep in range(num_episodes)]
    avg_reward = sum(m["total_reward"] for m in metrics) / len(metrics)
    avg_steps = sum(m["steps"] for m in metrics) / len(metrics)
    return {"num_episodes": len(metrics), "avg_reward": avg_reward, "avg_steps": avg_steps, "raw": metrics}


def main():
    """"
    run random agent script with:
        python3 /workspace/rl/random_agent.py --episodes [] --seed [] --max-steps [] --render --fps [] --verbose
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Create robot
    robot = Robot("assets/panda/panda.urdf")

    dataset = StateDataset.load_from_directory(
        robot=robot,
        directory="datasets/ae_aristotle1_5mm_cubbies",
        dataset_type=DatasetType.TRAIN,
        trajectory_key="global_solutions",
        num_robot_points=2048,
        num_obstacle_points=4096,
        num_target_points=128,
        random_scale=0.0,
        action_chunk_length=1,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("✓ DataLoader created")

    env = AvoidEverythingEnv(dataloader=dataloader, 
                             render_mode=RENDER_MODE,
                             render_fps=args.fps,
                             max_episode_steps=args.max_steps)

    policy = RandomPolicy(env.action_space, seed=args.seed)
    results = evaluate_policy(policy, env, num_episodes=args.episodes, render=args.render, verbose=args.verbose)
    print(f"\nRan {results['num_episodes']} random episodes — avg_reward={results['avg_reward']:.3f}, avg_steps={results['avg_steps']:.1f}")


if __name__ == "__main__":
    main()

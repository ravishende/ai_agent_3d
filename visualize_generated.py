"""
Generate a map and visualize its training metrics
Usage: python visualize_generated.py --length 50 --width 7 --difficulty hard
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from map_generator import MapGenerator
from gymnasium_env import ObstacleCourseEnv
from integrated_policy_iteration import IntegratedPolicyIteration
from core import INIT_GAME

# Set style
sns.set_theme(style="whitegrid", palette="husl")


def train_and_visualize(length, width, difficulty, gamma=0.98, save=False):
    """Generate a map, train on it, and visualize the results."""
    
    print("\n" + "="*70)
    print("GENERATING MAP")
    print("="*70)
    print(f"  Length: {length} timesteps")
    print(f"  Width: {width} lanes")
    print(f"  Difficulty: {difficulty}")
    print(f"  Gamma: {gamma}")
    
    # Generate map - FIX: Properly unpack the tuple
    generator = MapGenerator(n_cols=width)
    game_map, trap_cols = generator.generate_track(
        timesteps=length,
        trap_spawn_prob=0.5,
        difficulty=difficulty
    )
    
    # Verify map structure and fix if needed
    if len(game_map) == 0:
        print("ERROR: Generated empty map!")
        return None, None, None
    
    # Check if map elements are proper numpy arrays with correct shape (3, width)
    if isinstance(game_map[0], np.ndarray):
        expected_shape = (3, width)
        if game_map[0].shape != expected_shape:
            print(f"WARNING: Unexpected map shape {game_map[0].shape}, expected {expected_shape}")
            print("This might cause errors. Check your map_generator.py")
    
    actual_length = len(game_map)
    print(f"\n✓ Generated map with {actual_length} timesteps")
    print(f"  Each slice shape: {game_map[0].shape}")
    print("\nFirst 3 slices:")
    for i in range(min(3, len(game_map))):
        print(f"\nTimestep {i}:")
        print(game_map[i])
    
    # Train
    print("\n" + "="*70)
    print("TRAINING AGENT")
    print("="*70)
    
    # Initialize the game map in core (required for State objects)
    INIT_GAME(game_map, trap_cols)
    
    env = ObstacleCourseEnv(game_map)
    pi = IntegratedPolicyIteration(env, gamma=gamma)
    converged, iterations = pi.run(max_iterations=100, verbose=True)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    eval_result = pi.evaluate_policy_on_map()
    print(f"  Success: {'✓ Yes' if eval_result['success'] else '✗ No'}")
    print(f"  Total Reward: {eval_result['total_reward']}/{actual_length}")
    print(f"  Steps Taken: {eval_result['steps']}")
    print(f"  Action Sequence: {' → '.join([a.name for a in eval_result['actions'][:15]])}...")
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_training_metrics(pi, 
                         map_name=f"{actual_length}x{width} {difficulty.capitalize()} Map",
                         save_path=f"training_{actual_length}x{width}_{difficulty}.png" if save else None)
    
    return pi, eval_result, game_map


def plot_training_metrics(pi, map_name="Map", save_path=None):
    """Create comprehensive training visualizations."""
    metrics = pi.metrics
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Policy Iteration Training - {map_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Value Function Convergence
    ax = axes[0, 0]
    ax.plot(metrics['iteration'], metrics['mean_value'], 
            linewidth=2.5, label='Mean Value', marker='o', markersize=5, color='#2E86AB')
    ax.plot(metrics['iteration'], metrics['max_value'], 
            linewidth=2.5, label='Max Value', marker='s', markersize=5, alpha=0.7, color='#A23B72')
    ax.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Value', fontweight='bold', fontsize=11)
    ax.set_title('Value Function Convergence', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Policy Changes per Iteration
    ax = axes[0, 1]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(metrics['iteration'])))
    bars = ax.bar(metrics['iteration'], metrics['policy_changes'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of State Updates', fontweight='bold', fontsize=11)
    ax.set_title('Policy Changes per Iteration', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Policy Stability
    ax = axes[0, 2]
    ax.plot(metrics['iteration'], metrics['policy_stability'], 
            linewidth=2.5, color='#06A77D', marker='o', markersize=5)
    ax.axhline(y=1.0, color='#D62246', linestyle='--', linewidth=2, alpha=0.6, label='Full Stability')
    ax.fill_between(metrics['iteration'], metrics['policy_stability'], 1.0, 
                     alpha=0.2, color='#06A77D')
    ax.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Stability Ratio', fontweight='bold', fontsize=11)
    ax.set_title('Policy Stability Over Time', fontweight='bold', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Evaluation Iterations (Inner Loop)
    ax = axes[1, 0]
    ax.plot(metrics['iteration'], metrics['eval_iterations'], 
            linewidth=2.5, color='#8E44AD', marker='d', markersize=5)
    ax.fill_between(metrics['iteration'], metrics['eval_iterations'], 
                     alpha=0.2, color='#8E44AD')
    ax.set_xlabel('Policy Iteration (Outer Loop)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Evaluation Iterations (Inner Loop)', fontweight='bold', fontsize=11)
    ax.set_title('Policy Evaluation Convergence Speed', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 5. Bellman Error (Convergence Rate)
    ax = axes[1, 1]
    ax.semilogy(metrics['iteration'], metrics['mean_value_change'], 
                linewidth=2.5, color='#E67E22', marker='v', markersize=5)
    ax.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean Value Change (log scale)', fontweight='bold', fontsize=11)
    ax.set_title('Bellman Error (Convergence Rate)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # 6. Cumulative Policy Changes
    ax = axes[1, 2]
    cumulative_changes = np.cumsum(metrics['policy_changes'])
    ax.fill_between(metrics['iteration'], cumulative_changes, 
                     alpha=0.3, color='#3498DB')
    ax.plot(metrics['iteration'], cumulative_changes, 
            linewidth=2.5, color='#2C3E50', marker='o', markersize=5)
    ax.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cumulative Updates', fontweight='bold', fontsize=11)
    ax.set_title('Total Policy Changes Over Time', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    total_iters = len(metrics['iteration'])
    total_changes = sum(metrics['policy_changes'])
    final_stability = metrics['policy_stability'][-1]
    
    stats_text = (f"Training Summary:\n"
                 f"• Total Iterations: {total_iters}\n"
                 f"• Total Policy Updates: {total_changes}\n"
                 f"• Final Stability: {final_stability:.3f}\n"
                 f"• Final Mean Value: {metrics['mean_value'][-1]:.3f}")
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Policy Iterations: {total_iters}")
    print(f"Total Policy Changes: {total_changes}")
    print(f"Final Policy Stability: {final_stability:.4f}")
    print(f"Final Mean State Value: {metrics['mean_value'][-1]:.4f}")
    print(f"Final Max State Value: {metrics['max_value'][-1]:.4f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize training on a custom map"
    )
    
    parser.add_argument(
        "--length",
        type=int,
        default=50,
        help="Map length in timesteps (default: 50)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=7,
        help="Map width in lanes (default: 7)"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard", "expert"],
        help="Map difficulty (default: hard)"
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Discount factor (default: 0.98)"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the visualization to a file"
    )
    
    args = parser.parse_args()
    
    train_and_visualize(
        length=args.length,
        width=args.width,
        difficulty=args.difficulty,
        gamma=args.gamma,
        save=args.save
    )


if __name__ == "__main__":
    main()
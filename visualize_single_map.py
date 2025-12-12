"""
Visualize training metrics for a single map
Usage: python visualize_single_map.py --map 2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from maps import maps
from map_generator import MapGenerator
from gymnasium_env import ObstacleCourseEnv
from integrated_policy_iteration import IntegratedPolicyIteration

# Set style
sns.set_theme(style="whitegrid", palette="husl")


def train_and_collect_metrics(game_map, gamma=0.98):
    """Train on a map and return the policy iteration object with metrics."""
    env = ObstacleCourseEnv(game_map)
    pi = IntegratedPolicyIteration(env, gamma=gamma)
    pi.run(max_iterations=100, verbose=True)
    return pi


def plot_training_metrics(pi, map_name="Map", save_path=None):
    """Create a comprehensive visualization of training metrics."""
    metrics = pi.metrics
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Policy Iteration Training Metrics - {map_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Value Function Convergence
    ax = axes[0, 0]
    ax.plot(metrics['iteration'], metrics['mean_value'], 
            linewidth=2, label='Mean Value', marker='o', markersize=4)
    ax.plot(metrics['iteration'], metrics['max_value'], 
            linewidth=2, label='Max Value', marker='s', markersize=4, alpha=0.7)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Value Function Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Policy Changes per Iteration
    ax = axes[0, 1]
    bars = ax.bar(metrics['iteration'], metrics['policy_changes'], 
                   color=sns.color_palette("coolwarm", len(metrics['iteration'])))
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Number of State Policy Changes', fontweight='bold')
    ax.set_title('Policy Updates per Iteration', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Policy Stability
    ax = axes[0, 2]
    ax.plot(metrics['iteration'], metrics['policy_stability'], 
            linewidth=2, color='green', marker='o', markersize=4)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full Stability')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Stability Ratio', fontweight='bold')
    ax.set_title('Policy Stability (1 - changes/total_states)', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Evaluation Iterations
    ax = axes[1, 0]
    ax.plot(metrics['iteration'], metrics['eval_iterations'], 
            linewidth=2, color='purple', marker='d', markersize=4)
    ax.set_xlabel('Policy Iteration', fontweight='bold')
    ax.set_ylabel('Evaluation Iterations', fontweight='bold')
    ax.set_title('Policy Evaluation Convergence Speed', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. Mean Value Change (Bellman Error)
    ax = axes[1, 1]
    ax.semilogy(metrics['iteration'], metrics['mean_value_change'], 
                linewidth=2, color='orange', marker='v', markersize=4)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Mean Value Change (log scale)', fontweight='bold')
    ax.set_title('Convergence Rate (Bellman Error)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Cumulative Policy Changes
    ax = axes[1, 2]
    cumulative_changes = np.cumsum(metrics['policy_changes'])
    ax.fill_between(metrics['iteration'], cumulative_changes, 
                     alpha=0.4, color='skyblue')
    ax.plot(metrics['iteration'], cumulative_changes, 
            linewidth=2, color='blue', marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Cumulative Changes', fontweight='bold')
    ax.set_title('Total Policy Updates Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Iterations: {len(metrics['iteration'])}")
    print(f"Final Mean Value: {metrics['mean_value'][-1]:.4f}")
    print(f"Final Max Value: {metrics['max_value'][-1]:.4f}")
    print(f"Total Policy Changes: {sum(metrics['policy_changes'])}")
    print(f"Final Policy Stability: {metrics['policy_stability'][-1]:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training metrics for a specific map"
    )
    
    parser.add_argument(
        "--map",
        type=int,
        default=1,
        help=f"Which map to visualize (1-{len(maps)})"
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
        help="Save the figure to a file"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate visualizations for all maps"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Generate for all maps
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS FOR ALL MAPS")
        print("="*60)
        
        for i in range(len(maps)):
            map_num = i + 1
            print(f"\n\nProcessing Map {map_num}...")
            print("-" * 60)
            
            pi = train_and_collect_metrics(maps[i], gamma=args.gamma)
            
            save_path = f"training_metrics_map{map_num}.png" if args.save else None
            plot_training_metrics(
                pi, 
                map_name=f"Map {map_num} (Length={len(maps[i])} steps)",
                save_path=save_path
            )
    else:
        # Generate for single map
        map_idx = args.map - 1
        
        if not (0 <= map_idx < len(maps)):
            print(f"Error: Map must be between 1 and {len(maps)}")
            return
        
        game_map = maps[map_idx]
        
        print("\n" + "="*60)
        print(f"TRAINING ON MAP {args.map}")
        print("="*60)
        print(f"Map length: {len(game_map)} timesteps")
        print(f"Gamma: {args.gamma}")
        print("="*60 + "\n")
        
        pi = train_and_collect_metrics(game_map, gamma=args.gamma)
        
        save_path = f"training_metrics_map{args.map}.png" if args.save else None
        plot_training_metrics(
            pi,
            map_name=f"Map {args.map} (Length={len(game_map)} steps)",
            save_path=save_path
        )


if __name__ == "__main__":
    main()

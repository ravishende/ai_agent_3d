"""
Visualization script for Policy Iteration training metrics
Usage: python visualize_training.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from maps import maps
from map_generator import MapGenerator
from gymnasium_env import ObstacleCourseEnv
from integrated_policy_iteration import IntegratedPolicyIteration
import json

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def train_and_collect_metrics(game_map, gamma=0.98):
    """Train on a map and return the policy iteration object with metrics."""
    env = ObstacleCourseEnv(game_map)
    pi = IntegratedPolicyIteration(env, gamma=gamma)
    pi.run(max_iterations=100, verbose=False)
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
        print(f"Saved figure to {save_path}")
    
    plt.show()


def compare_multiple_runs(game_maps, map_names, gamma=0.98, save_path=None):
    """Compare training across multiple maps."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison Across Different Maps', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(game_maps))
    
    for idx, (game_map, name) in enumerate(zip(game_maps, map_names)):
        print(f"Training on {name}...")
        pi = train_and_collect_metrics(game_map, gamma)
        metrics = pi.metrics
        color = colors[idx]
        
        # Mean Value Convergence
        axes[0, 0].plot(metrics['iteration'], metrics['mean_value'], 
                       linewidth=2, label=name, color=color, marker='o', markersize=3)
        
        # Policy Changes
        axes[0, 1].plot(metrics['iteration'], metrics['policy_changes'], 
                       linewidth=2, label=name, color=color, marker='s', markersize=3)
        
        # Policy Stability
        axes[1, 0].plot(metrics['iteration'], metrics['policy_stability'], 
                       linewidth=2, label=name, color=color, marker='^', markersize=3)
        
        # Evaluation Iterations
        axes[1, 1].plot(metrics['iteration'], metrics['eval_iterations'], 
                       linewidth=2, label=name, color=color, marker='d', markersize=3)
    
    axes[0, 0].set_title('Mean Value Function', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Policy Changes per Iteration', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Number of Changes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Policy Stability', fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Stability Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Evaluation Iterations', fontweight='bold')
    axes[1, 1].set_xlabel('Policy Iteration')
    axes[1, 1].set_ylabel('Eval Iterations')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison figure to {save_path}")
    
    plt.show()


def analyze_difficulty_scaling(difficulties=['easy', 'medium', 'hard', 'expert'], 
                               map_length=15, map_width=5, save_path=None):
    """Analyze how training metrics scale with difficulty."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics vs Map Difficulty', fontsize=16, fontweight='bold')
    
    results = {
        'difficulty': [],
        'convergence_iterations': [],
        'total_policy_changes': [],
        'final_mean_value': [],
        'success_rate': []
    }
    
    for difficulty in difficulties:
        print(f"\nGenerating and training on {difficulty} map...")
        generator = MapGenerator(n_cols=map_width)
        game_map = generator.generate_track(timesteps=map_length, difficulty=difficulty)
        
        pi = train_and_collect_metrics(game_map, gamma=0.98)
        metrics = pi.metrics
        
        results['difficulty'].append(difficulty)
        results['convergence_iterations'].append(len(metrics['iteration']))
        results['total_policy_changes'].append(sum(metrics['policy_changes']))
        results['final_mean_value'].append(metrics['mean_value'][-1])
        
        # Test success rate
        eval_result = pi.evaluate_policy_on_map()
        results['success_rate'].append(1.0 if eval_result['success'] else 0.0)
    
    # Plot 1: Convergence Iterations
    ax = axes[0, 0]
    bars = ax.bar(results['difficulty'], results['convergence_iterations'], 
                   color=sns.color_palette("rocket", len(difficulties)))
    ax.set_xlabel('Difficulty', fontweight='bold')
    ax.set_ylabel('Iterations to Converge', fontweight='bold')
    ax.set_title('Convergence Speed vs Difficulty', fontweight='bold')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Total Policy Changes
    ax = axes[0, 1]
    bars = ax.bar(results['difficulty'], results['total_policy_changes'], 
                   color=sns.color_palette("mako", len(difficulties)))
    ax.set_xlabel('Difficulty', fontweight='bold')
    ax.set_ylabel('Total Policy Updates', fontweight='bold')
    ax.set_title('Learning Complexity vs Difficulty', fontweight='bold')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Final Mean Value
    ax = axes[1, 0]
    bars = ax.bar(results['difficulty'], results['final_mean_value'], 
                   color=sns.color_palette("viridis", len(difficulties)))
    ax.set_xlabel('Difficulty', fontweight='bold')
    ax.set_ylabel('Final Mean State Value', fontweight='bold')
    ax.set_title('Expected Reward vs Difficulty', fontweight='bold')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Success Rate
    ax = axes[1, 1]
    bars = ax.bar(results['difficulty'], results['success_rate'], 
                   color=['green' if x == 1.0 else 'red' for x in results['success_rate']])
    ax.set_xlabel('Difficulty', fontweight='bold')
    ax.set_ylabel('Success Rate', fontweight='bold')
    ax.set_title('Agent Success vs Difficulty', fontweight='bold')
    ax.set_ylim([0, 1.1])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = '✓ Success' if height == 1.0 else '✗ Failed'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved difficulty analysis to {save_path}")
    
    plt.show()
    
    return results


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("POLICY ITERATION TRAINING VISUALIZATION")
    print("=" * 60)
    
    # 1. Single map detailed analysis
    # 1. Single map detailed analysis for all maps
    for i in range(len(maps)):
        print(f"\n1. Training on Map {i+1}...")
        pi = train_and_collect_metrics(maps[i], gamma=0.98)
        plot_training_metrics(pi, map_name=f"Map {i+1} (Length={len(maps[i])})", 
                            save_path=f"training_metrics_map{i+1}.png")
    
    # 2. Compare predefined maps
    print("\n2. Comparing all predefined maps...")
    map_names = [f"Map {i+1} (len={len(m)})" for i, m in enumerate(maps)]
    compare_multiple_runs(maps, map_names, gamma=0.98, 
                         save_path="training_comparison.png")
    
    # 3. Difficulty scaling analysis
    print("\n3. Analyzing difficulty scaling...")
    difficulty_results = analyze_difficulty_scaling(
        difficulties=['easy', 'medium', 'hard'],
        map_length=15,
        map_width=5,
        save_path="difficulty_analysis.png"
    )
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(difficulty_results, f, indent=2)
    print("\nSaved difficulty results to training_results.json")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Compare convergence across multiple maps with different complexities
Usage: python compare_maps.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from map_generator import MapGenerator
from gymnasium_env import ObstacleCourseEnv
from integrated_policy_iteration import IntegratedPolicyIteration
from core import INIT_GAME

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (20, 12)

# Create output directory for visualizations
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_on_map(length, width, difficulty, gamma=0.98):
    """Generate a map and train on it, returning metrics."""
    print(f"\n{'='*70}")
    print(f"Training on {length}x{width} {difficulty.upper()} map...")
    print(f"{'='*70}")
    
    # Generate map
    generator = MapGenerator(n_cols=width)
    game_map, trap_cols = generator.generate_track(
        timesteps=length,
        trap_spawn_prob=0.5,
        difficulty=difficulty
    )
    
    actual_length = len(game_map)
    print(f"Generated map: {actual_length} timesteps, {width} lanes")
    
    # Initialize and train
    INIT_GAME(game_map, trap_cols)
    env = ObstacleCourseEnv(game_map)
    pi = IntegratedPolicyIteration(env, gamma=gamma)
    converged, iterations = pi.run(max_iterations=100, verbose=False)
    
    # Evaluate
    eval_result = pi.evaluate_policy_on_map()
    success_status = "✓ SUCCESS" if eval_result['success'] else "✗ FAILED"
    print(f"Result: {success_status} | Iterations: {iterations} | Reward: {eval_result['total_reward']}/{actual_length}")
    
    return {
        'pi': pi,
        'metrics': pi.metrics,
        'eval': eval_result,
        'length': actual_length,
        'width': width,
        'difficulty': difficulty,
        'converged': converged,
        'iterations': iterations
    }


def compare_convergence_across_maps():
    """Train on progressively harder maps and visualize comparison."""
    
    print("\n" + "="*70)
    print("MULTI-MAP CONVERGENCE COMPARISON")
    print("="*70)
    
    # Define map configurations
    map_configs = [
        {'length': 10, 'width': 3, 'difficulty': 'easy', 'gamma': 0.98},
        {'length': 25, 'width': 5, 'difficulty': 'medium', 'gamma': 0.98},
        {'length': 50, 'width': 7, 'difficulty': 'hard', 'gamma': 0.98},
        {'length': 100, 'width': 9, 'difficulty': 'expert', 'gamma': 0.98}
    ]
    
    # Train on all maps
    results = []
    for config in map_configs:
        result = train_on_map(**config)
        results.append(result)
    
    # Create comprehensive visualizations
    create_comparison_plots(results)
    create_summary_statistics(results)
    
    return results


def create_comparison_plots(results):
    """Create comprehensive comparison visualizations."""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color palette for different maps
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    map_labels = [
        f"{r['length']}x{r['width']} {r['difficulty'].capitalize()}" 
        for r in results
    ]
    
    # 1. Value Function Convergence (Top Left - Large)
    ax1 = fig.add_subplot(gs[0, :2])
    for idx, (result, color, label) in enumerate(zip(results, colors, map_labels)):
        metrics = result['metrics']
        ax1.plot(metrics['iteration'], metrics['mean_value'], 
                linewidth=3, label=label, color=color, marker='o', 
                markersize=6, alpha=0.8)
    ax1.set_xlabel('Policy Iteration', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Mean State Value', fontweight='bold', fontsize=13)
    ax1.set_title('Value Function Convergence Across Map Complexities', 
                  fontweight='bold', fontsize=15)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Policy Stability Comparison (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    for idx, (result, color, label) in enumerate(zip(results, colors, map_labels)):
        metrics = result['metrics']
        ax2.plot(metrics['iteration'], metrics['policy_stability'], 
                linewidth=2.5, color=color, alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.4)
    ax2.set_xlabel('Iteration', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Stability Ratio', fontweight='bold', fontsize=11)
    ax2.set_title('Policy Stability', fontweight='bold', fontsize=13)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    
    # 3. Total Policy Changes (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    total_changes = [sum(r['metrics']['policy_changes']) for r in results]
    bars = ax3.bar(range(len(results)), total_changes, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([f"{r['length']}x{r['width']}\n{r['difficulty']}" 
                          for r in results], fontsize=9)
    ax3.set_ylabel('Total Policy Updates', fontweight='bold', fontsize=11)
    ax3.set_title('Learning Complexity', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, total_changes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 4. Convergence Speed (Middle Center)
    ax4 = fig.add_subplot(gs[1, 1])
    iterations = [r['iterations'] for r in results]
    bars = ax4.bar(range(len(results)), iterations, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels([f"{r['length']}x{r['width']}\n{r['difficulty']}" 
                          for r in results], fontsize=9)
    ax4.set_ylabel('Iterations to Converge', fontweight='bold', fontsize=11)
    ax4.set_title('Convergence Speed', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 5. Success Rate (Middle Right)
    ax5 = fig.add_subplot(gs[1, 2])
    success = [1 if r['eval']['success'] else 0 for r in results]
    bar_colors = ['green' if s == 1 else 'red' for s in success]
    bars = ax5.bar(range(len(results)), success, color=bar_colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    ax5.set_xticks(range(len(results)))
    ax5.set_xticklabels([f"{r['length']}x{r['width']}\n{r['difficulty']}" 
                          for r in results], fontsize=9)
    ax5.set_ylabel('Success', fontweight='bold', fontsize=11)
    ax5.set_title('Agent Success Rate', fontweight='bold', fontsize=13)
    ax5.set_ylim([0, 1.2])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Failed', 'Success'])
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, success):
        label = '✓' if s == 1 else '✗'
        ax5.text(bar.get_x() + bar.get_width()/2., s + 0.05,
                label, ha='center', va='bottom', 
                fontweight='bold', fontsize=20)
    
    # 6. Bellman Error Convergence (Bottom Left)
    ax6 = fig.add_subplot(gs[2, 0])
    for idx, (result, color, label) in enumerate(zip(results, colors, map_labels)):
        metrics = result['metrics']
        ax6.semilogy(metrics['iteration'], metrics['mean_value_change'], 
                    linewidth=2.5, color=color, marker='v', 
                    markersize=5, alpha=0.7)
    ax6.set_xlabel('Iteration', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Mean Value Change (log)', fontweight='bold', fontsize=11)
    ax6.set_title('Bellman Error Decay', fontweight='bold', fontsize=13)
    ax6.grid(True, alpha=0.3, which='both')
    
    # 7. Evaluation Iterations (Bottom Center)
    ax7 = fig.add_subplot(gs[2, 1])
    for idx, (result, color, label) in enumerate(zip(results, colors, map_labels)):
        metrics = result['metrics']
        ax7.plot(metrics['iteration'], metrics['eval_iterations'], 
                linewidth=2.5, color=color, marker='d', 
                markersize=5, alpha=0.7)
    ax7.set_xlabel('Policy Iteration', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Eval Iterations', fontweight='bold', fontsize=11)
    ax7.set_title('Policy Evaluation Speed', fontweight='bold', fontsize=13)
    ax7.grid(True, alpha=0.3)
    
    # 8. State Space Complexity (Bottom Right)
    ax8 = fig.add_subplot(gs[2, 2])
    state_space_sizes = [r['pi'].n_states for r in results]
    bars = ax8.bar(range(len(results)), state_space_sizes, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax8.set_xticks(range(len(results)))
    ax8.set_xticklabels([f"{r['length']}x{r['width']}\n{r['difficulty']}" 
                          for r in results], fontsize=9)
    ax8.set_ylabel('Number of States', fontweight='bold', fontsize=11)
    ax8.set_title('State Space Size', fontweight='bold', fontsize=13)
    ax8.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, state_space_sizes):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Add overall title
    fig.suptitle('Policy Iteration: Multi-Map Convergence Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Save and show
    output_path = os.path.join(OUTPUT_DIR, 'multi_map_convergence_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive comparison to '{output_path}'")
    plt.show()


def create_summary_statistics(results):
    """Print detailed summary statistics."""
    
    print("\n" + "="*70)
    print("DETAILED SUMMARY STATISTICS")
    print("="*70)
    
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result['length']}x{result['width']} {result['difficulty'].upper()} MAP")
        print("-" * 70)
        print(f"  State Space Size:      {result['pi'].n_states:,} states")
        print(f"  Convergence:           {result['iterations']} iterations")
        print(f"  Total Policy Changes:  {sum(result['metrics']['policy_changes']):,}")
        print(f"  Final Mean Value:      {result['metrics']['mean_value'][-1]:.4f}")
        print(f"  Final Max Value:       {result['metrics']['max_value'][-1]:.4f}")
        print(f"  Final Stability:       {result['metrics']['policy_stability'][-1]:.4f}")
        print(f"  Success:               {'✓ YES' if result['eval']['success'] else '✗ NO'}")
        print(f"  Total Reward:          {result['eval']['total_reward']}/{result['length']}")
        
        # Complexity metrics
        avg_eval_iters = np.mean(result['metrics']['eval_iterations'])
        total_eval_iters = sum(result['metrics']['eval_iterations'])
        print(f"  Avg Eval Iterations:   {avg_eval_iters:.1f}")
        print(f"  Total Eval Iterations: {total_eval_iters:,}")
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    # Complexity scaling
    print("\nComplexity Scaling:")
    for idx in range(1, len(results)):
        prev = results[idx-1]
        curr = results[idx]
        
        state_ratio = curr['pi'].n_states / prev['pi'].n_states
        iter_ratio = curr['iterations'] / prev['iterations']
        
        print(f"  {prev['difficulty']} → {curr['difficulty']}:")
        print(f"    State space: {state_ratio:.2f}x larger")
        print(f"    Convergence: {iter_ratio:.2f}x {'slower' if iter_ratio > 1 else 'faster'}")
    
    # Success analysis
    print("\nSuccess Analysis:")
    success_count = sum(1 for r in results if r['eval']['success'])
    print(f"  Total Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    results = compare_convergence_across_maps()
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nVisualizations saved to: ./{OUTPUT_DIR}/")
    print(f"  - multi_map_convergence_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

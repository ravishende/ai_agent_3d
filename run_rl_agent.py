"""
"""
import argparse
import numpy as np
from maps import maps
from map_generator import MapGenerator
from integrated_policy_iteration import solve_map
from process_actions import get_locations_list
from env_3d import visualize
from core import INIT_GAME


def run_rl_on_map(game_map, trap_cols=None, trap_prob=0.3, gamma=0.98, visualize_result=False, 
                  cube_size=2.0, spacing=40.0, verbose=True):
    if verbose:
        print("="*60)
        print("REINFORCEMENT LEARNING AGENT")
        print("="*60)
        print(f"Map length: {len(game_map)} timesteps")
        print(f"Discount factor (gamma): {gamma}")
        print()

    INIT_GAME(game_map, trap_cols, trap_prob)
    
    optimal_actions, policy_iter = solve_map(game_map, gamma=gamma, verbose=verbose)
    
    results = policy_iter.evaluate_policy_on_map()
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Total Reward: {results['total_reward']}")
        print(f"Steps Taken: {results['steps']}")
        print(f"Map Length: {len(game_map)}")
        
        if results['success']:
            print("\n✓ Agent successfully completed the course!")
        else:
            print("\n✗ Agent crashed during the course.")
    
    if visualize_result:
        if verbose:
            print("\nLaunching 3D visualization...")
        locations = get_locations_list(optimal_actions, map_width=game_map[0].shape[1])
        visualize(game_map=game_map,
                  player_locations=locations,
                  trap_cols=trap_cols,
                  cube_size=cube_size,
                  spacing=spacing)
    
    
    
    return {
        'actions': optimal_actions,
        'action_names': [a.name for a in optimal_actions],
        'success': results['success'],
        'total_reward': results['total_reward'],
        'steps': results['steps'],
    }


def main():
    """Main entry point with command-line interface."""
    args = get_args()
    game_map = []
    trap_cols = []

    if args.fixed:
        map_idx = args.map - 1
        if not (0 <= map_idx < len(maps)):
            print(f"Error: Map index must be between 1 and {len(maps)}")
            return
        game_map = maps[map_idx]
        if not args.quiet:
            print(f"Using fixed map {args.map}")
    else:
        if not args.quiet:
            print(f"Generating {args.difficulty} map with {args.length} timesteps...")
        generator = MapGenerator(n_cols=args.width)
        game_map, trap_cols = generator.generate_track(
            timesteps=args.length,
            trap_spawn_prob=args.trap_spawn_prob,
            difficulty=args.difficulty
        )

    results = run_rl_on_map(
        game_map,
        trap_cols=trap_cols,
        trap_prob=args.trap_death_prob,
        gamma=args.gamma,
        visualize_result=args.visualize,
        cube_size=args.cubesize,
        spacing=args.spacing,
        verbose=not args.quiet
    )

    if not args.quiet:
        print("\n" + "="*60)
        print("FINAL ACTION SEQUENCE")
        print("="*60)
        print(" → ".join(results['action_names']))
        print()



def test_all_maps():
    """Test the RL agent on all predefined maps."""
    print("\n" + "="*70)
    print("TESTING RL AGENT ON ALL PREDEFINED MAPS")
    print("="*70 + "\n")
    
    for i, game_map in enumerate(maps, 1):
        print(f"\n{'*'*70}")
        print(f"MAP {i} (Length: {len(game_map)} timesteps)")
        print('*'*70)
        
        results = run_rl_on_map(game_map, verbose=True, visualize_result=False)
        
        if results['success']:
            print(f"✓ Map {i}: SUCCESS")
        else:
            print(f"✗ Map {i}: FAILED")

def get_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent to navigate obstacle course"
    )
    
    parser.add_argument(
        "--map",
        type=int,
        default=1,
        help=f"Which fixed map to use (1-{len(maps)})"
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use a fixed map instead of generating one"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty of generated map"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="Length of generated map (timesteps)"
    )

    parser.add_argument(
        "--width", 
        type=int,
        default=3,
        help="Width of the generated track (default: 3)"
    )
    parser.add_argument(
        "--trap_death_prob", "--trap-death-prob", "--trapdeathprob", 
        type=float,
        default=0.3,
        help="Probability of a trap activating when the player steps on it -- 0 <= trap_prob < 1"
    )
    parser.add_argument(
        "--trap_spawn_prob", "--trap-spawn-prob", "--trapspawnprob",
        type=float,
        default=0.3,
        help="Probability of traps spawning in any given slice (controls how many traps there are)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Discount factor for RL (default: 0.98)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show 3D visualization of the solution"
    )
    parser.add_argument(
        "--cubesize", "--cube-size", "--cube_size",
        type=float,
        default=2.0,
        help="Size of cubes in visualization"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=40.0,
        help="Spacing between slices in visualization"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()


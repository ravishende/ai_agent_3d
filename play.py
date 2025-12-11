"""
Play a text based version of the game and then see your actions visualized in 3D
"""

import argparse
from maps import maps
from play_text import play_text
from process_actions import get_locations_list
from env_3d import visualize
from map_generator import MapGenerator

def main():
    args = get_args()
    game_map = []
    trap_cols = []
    if not args.fixed:
        # OPTION A: Generate a Random Map
        print(f"Generating {args.difficulty} 3x{args.width} map with {args.length} steps...")
        generator = MapGenerator(n_cols=args.width)
        game_map, trap_cols = generator.generate_track(
            timesteps=args.length, trap_spawn_prob=args.trap_spawn_prob, difficulty=args.difficulty)
    else:
        # OPTION B: Use a Fixed Map
        map_idx = args.map - 1
        valid_map_choice = 0 <= map_idx < len(maps)
        if not valid_map_choice:
            print(f"\n\nPlease choose a map between 1 and {len(maps)}. You chose {args.map}\n\n")
            return
        game_map = maps[map_idx]
        trap_cols = [-1 for _ in range(len(game_map))]
    
    moves = play_text(game_map,
                      trap_cols,
                      trap_death_prob=args.trap_death_prob,
                      print_moves=False,
                      difficulty=args.difficulty)
    locations = get_locations_list(actions_list=moves, map_width=args.width)
    visualize(game_map, trap_cols=trap_cols, player_locations=locations, cube_size=args.cubesize, spacing=args.spacing)



def get_args():
    """Get Command Line Arguments for map generation and 3D visualization"""
    parser = argparse.ArgumentParser(description="parser for map generation and visualization")
    
    parser.add_argument(
        "--map",
        type=int,
        default=1,
        help=f"Which map to choose - from 1 to {len(maps)} (default: 1)"
    )
    parser.add_argument(
        "--fixed", 
        default=False,
        action="store_true",
        help="Use a fixed map instead of generating a random one"
    )
    difficulty_choices = ["easy", "medium", "hard", "expert"]
    parser.add_argument(
        "--difficulty", 
        type=str,
        default="medium",
        choices=difficulty_choices,
        help=f"Difficulty of generated map {difficulty_choices}"
    )
    parser.add_argument(
        "--length", 
        type=int,
        default=50,
        help="Length of the generated track (default: 50)"
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
        "--cubesize", "--cube-size", "--cube_size",
        type=float,
        default=2.0,
        help="size of cubes to visualize (default: 2.0)"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=40.0,
        help="spacing of slices of cubes in visualization (default: 40.0)"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
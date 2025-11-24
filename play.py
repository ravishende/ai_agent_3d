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
    if not args.fixed:
        # OPTION A: Generate a Random Map
        print(f"Generating a {args.difficulty} map with {args.length} steps...")
        generator = MapGenerator()
        game_map = generator.generate_track(timesteps=args.length, difficulty=args.difficulty)
    else:
        # OPTION B: Use a Fixed Map
        map_idx = args.map - 1
        valid_map_choice = 0 <= map_idx < len(maps)
        if not valid_map_choice:
            print(f"\n\nPlease choose a map between 1 and {len(maps)}. You chose {args.map}\n\n")
            return
        game_map = maps[map_idx]
    
    moves = play_text(game_map, print_moves=False)
    locations = get_locations_list(actions_list=moves)
    visualize(game_map, player_locations=locations)



def get_args():
    """Get Command Line Arguments for map"""
    parser = argparse.ArgumentParser(description="Example parser with --map flag")
    
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
    parser.add_argument(
        "--difficulty", 
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Difficulty of generated map (easy, medium, hard)"
    )
    parser.add_argument(
        "--length", 
        type=int,
        default=50,
        help="Length of the generated track (default: 50)"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
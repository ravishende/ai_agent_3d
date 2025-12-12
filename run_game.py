import time
import argparse
from core import INIT_GAME
from env_3d import visualize
from map_generator import MapGenerator
from integrated_policy_iteration import solve_map
from map_io import save_map, load_map

def run_simulation(args):
    """
    Orchestrates the simulation with support for Loading and Saving maps.
    """
    print(f"\n{'='*60}")
    print(f" STARTING SIMULATION")
    if args.load:
        print(f"   Mode: LOADING map '{args.load}'")
    else:
        print(f"   Mode: GENERATING ({args.length} steps, {args.lanes} lanes, '{args.difficulty}')")
    print(f"{'='*60}")

    game_map = []
    trap_cols = []

    # --- 1. Map Acquisition (Generate or Load) ---
    if args.load:
        try:
            game_map, trap_cols = load_map(args.load)
        except Exception as e:
            print(f" Error loading map: {e}")
            return
    else:
        print("\n[1/4]  Generating Procedural Map...")
        generator = MapGenerator(n_cols=args.lanes)
        game_map, trap_cols = generator.generate_track(
            timesteps=args.length, 
            trap_spawn_prob=args.traps,
            difficulty=args.difficulty
        )

    # --- 1b. Save Map (Optional) ---
    if args.save:
        save_map(game_map, trap_cols, args.save)

    # --- 2. Initialize Game Core ---
    # This sets global variables in core.py necessary for physics/logic
    INIT_GAME(game_map, trap_cols, trap_death_prob=0.3)
    
    # --- 3. Solve Map ---
    print("\n[2/4]  Solving with Policy Iteration...")
    start_time = time.time()
    
    try:
        # solve_map creates the environment and runs the solver
        actions, policy_instance = solve_map(game_map, verbose=True)
    except Exception as e:
        print(f" Error during solving: {e}")
        return

    solve_time = time.time() - start_time
    print(f"       Solved in {solve_time:.4f} seconds")

    # --- 4. Extract Visual Path ---
    print("\n[3/4]  Calculating Trajectory...")
    # We need to run the optimal policy through the environment to get exact coordinates for the viewer
    env = policy_instance.env
    path_locations = []
    
    # Reset env to start
    obs, _ = env.reset()
    state = env._obs_to_state(obs)
    
    # Add initial standing position (Row 1 is standing)
    path_locations.append((1, state.player_col))
    
    total_reward = 0
    
    for i, action_enum in enumerate(actions):
        # Convert enum to index for Gym
        action_idx = env.ACTION_TO_IDX[action_enum]
        
        # Execute step
        obs, reward, terminated, truncated, _ = env.step(action_idx)
        total_reward += reward
        
        next_state = env._obs_to_state(obs)
        
        # --- MAPPING ACTIONS TO VISUAL ROWS ---
        # core.py Actions are defined as (row_delta, col_delta)
        # Standard Standing Row is 1.
        # JUMP (-1, 0) -> Visual Row 0
        # DUCK (1, 0)  -> Visual Row 2
        # LEFT/RIGHT/STAY (0, x) -> Visual Row 1
        d_row, _ = action_enum.value
        vis_row = 1 + d_row 
        
        # Use the column from the *next* state
        vis_col = next_state.player_col 
        
        if next_state.is_terminal():
            if reward == 0:
                print(f"       Agent predicted to DIE at step {i}")
            break
            
        path_locations.append((vis_row, vis_col))

    print(f"      🏆 Expected Reward: {total_reward}")
    
    # --- 5. Visualize ---
    print("\n[4/4]  Launching OpenGL Visualization...")
    print("      (Press ESC or close window to exit)")
    
    visualize(
        game_map=game_map,
        player_locations=path_locations,
        trap_cols=trap_cols,
        cube_size=2.0,
        spacing=40.0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Obstacle Course AI Runner")
    
    # Map Configuration
    parser.add_argument("--lanes", type=int, default=3, help="Number of lanes (columns)")
    parser.add_argument("--length", type=int, default=50, help="Length of the map")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--traps", type=float, default=0.5, help="Probability of trap spawning")
    
    # Save/Load Configuration
    parser.add_argument("--save", type=str, help="Filename to SAVE the generated map to (e.g., 'run1')")
    parser.add_argument("--load", type=str, help="Filename to LOAD a map from (e.g., 'run1') - overrides generation settings")

    args = parser.parse_args()
    
    run_simulation(args)
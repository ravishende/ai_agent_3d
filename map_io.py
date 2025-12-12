import os
import numpy as np

MAP_DIR = "maps"

def ensure_map_dir():
    """Creates the maps/ directory if it doesn't exist."""
    if not os.path.exists(MAP_DIR):
        os.makedirs(MAP_DIR)

def save_map(game_map: list[np.ndarray], trap_cols: list[int], name: str):
    """Saves the map and trap columns to maps/<name>.npz"""
    ensure_map_dir()
    
    # Ensure extension exists
    if not name.endswith('.npz'):
        name += '.npz'
        
    path = os.path.join(MAP_DIR, name)
    
    # Convert list of 2D arrays -> Single 3D array for efficient storage
    # Shape becomes (Timesteps, Rows, Cols)
    map_stack = np.array(game_map)
    trap_array = np.array(trap_cols)
    
    np.savez_compressed(path, map_stack=map_stack, trap_array=trap_array)
    print(f"Map saved to: {path}")

def load_map(name: str) -> tuple[list[np.ndarray], list[int]]:
    """
    Loads a map from maps/<name>.npz
    """
    # Handle extension
    if not name.endswith('.npz'):
        name += '.npz'
        
    path = os.path.join(MAP_DIR, name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find map file: {path}")
    
    try:
        data = np.load(path)
        map_stack = data['map_stack']
        trap_array = data['trap_array']
        
        # Convert 3D array back to list of 2D arrays
        game_map = [slice_2d for slice_2d in map_stack]
        trap_cols = trap_array.tolist()
        
        print(f"Map loaded from: {path}")
        print(f"Dimensions: {len(game_map)} steps, {game_map[0].shape[1]} lanes")
        
        return game_map, trap_cols
        
    except KeyError:
        raise ValueError(f"File {path} is not a valid map file (missing keys).")
    except Exception as e:
        raise RuntimeError(f"Failed to load map: {e}")
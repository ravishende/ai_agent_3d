"""
Functions to help with bitmap operations.

For a single grid in a 3xN map:
Bitmaps are represented as integers where the least significant bit is the arr[0][0]
and then the next least significant bits are arr[1][0], arr[2][0], then arr[0][1] and so on.

Essentially, it starts the least significant bit at [0,0] and then counts down in column order
This was done instead of row order because the number of rows (3) never changes in a slice, 
but slices can be 3xN, so the number of columns can vary.

Bit layout:
    (row=0, col=0) is LSB (bit 0)
    (row=1, col=0) is bit 1
    ...
    (row=n_rows-1, col=0) is bit (n_rows-1)
    (row=0, col=1) is bit (n_rows)
    ...
    (row=n_rows-1, col=map_width-1) is bit (n_rows*map_width - 1)
In this case, n_rows=3
"""
import random
import numpy as np

def array_to_bitmap(arr: np.ndarray) -> int:
    """
    Convert a numpy array to a bitmap (int).
    """
    bitmap = 0
    n_rows, n_cols = arr.shape
    for row in range(n_rows):
        for col in range(n_cols):
            bitmap |= int(arr[row,col]) << (col*n_rows + row)
    return bitmap

def bitmap_to_array(grid_bitmap:int, map_width:int, n_rows:int = 3) -> np.ndarray:
    """
    Given a grid bitmap and number of columns and rows, return a resulting array.
    """
    arr = np.ones((n_rows, map_width))
    for row in range(n_rows):
        for col in range(map_width):
            digit = (1 << (col*n_rows + row)) & grid_bitmap
            arr[row, col] = 1 if digit > 0 else 0
    return arr.astype(int)

def game_map_bitmaps_to_arrays(game_map:list[int], map_width:int, n_rows:int = 3) -> list[np.ndarray]:
    """
    Convert a game map to be a list of 2D arrays from of a list of bitmaps
    Parameters:
        game_map: gamemap as a list of bitmaps
        map_width: number of columns in a grid of the map
        num_rows: number of rows in a grid of the map
    """
    if not isinstance(game_map[0], int):
        err = f"elements of game_map expected to be type int but were {type(game_map[0])}"
        raise TypeError(err)
    assert map_width > 0, "map_width (number of columns per grid) must be positive"
    return [bitmap_to_array(grid_bitmap=grid, map_width=map_width, n_rows=n_rows) for grid in game_map]


def game_map_arrays_to_bitmaps(game_map:list[np.ndarray]) -> list[int]:
    """
    Convert a game map to be a list of bitmaps from of a list of 2D arrays
    Parameters:
        game_map: gamemap as a list of numpy 2D arrays
    """
    if not isinstance(game_map[0], np.ndarray):
        err = f"elements of game_map expected to be 2D numpy arrays but were {type(game_map[0])}"
        raise TypeError(err)
    return [array_to_bitmap(grid) for grid in game_map]



def obstacle_at(location: tuple[int,int], grid_bitmap:int, n_rows:int = 3):
    """
    Returns True if there is an obstacle (1) in the grid_bitmap at the given location (row,col)
    Parameters:
        location: (row_index, col_index) of the player
        grid_bitmap: bitmap representation of the specified grid in the game map
        n_rows: number of rows in the game map
    """
    row, col = location
    bit_location = col * n_rows + row
    return (grid_bitmap & (1 << bit_location)) > 0

def create_funnel_pattern(safe_col:int, n_rows:int, map_width:int) -> int:
    """
    Create a grid bitmap of a funnel pattern: 1's in all locations except all 0's in specified col
    Parameters:
        safe_col: index of column that should not have obstacles
        map_width: number of columns in a grid in the game map
        n_rows: number of rows in a grid in the game map
    returns:
        grid_bitmap: a bitmap of the grid with a funnel pattern
    """
    total_bits = n_rows*map_width
    all_ones = (1 << total_bits) - 1
    col_mask = 0b111 << (n_rows * safe_col)
    return all_ones & ~col_mask

def create_random_grid(obstacle_frequency:float, n_rows:int, map_width:int) -> int:
    """
    Create a grid bitmap of random noise generation with a specified obstacle frequency
    Parameters:
        obstacle_frequency: a number between 0 and 1 representing how often obstacles should appear
        map_width: number of columns in a grid in the game map
        n_rows: number of rows in a grid in the game map
    returns:
        grid_bitmap: a randomized bitmap of the grid with a given obstacle frequency
    """
    grid_bitmap = 0
    for col in range(map_width):
        base = col * n_rows
        for row in range(n_rows):
            if random.random() < obstacle_frequency:
                # add obstacle --> bit is 1
                grid_bitmap |= (1 << (base + row))
    return grid_bitmap
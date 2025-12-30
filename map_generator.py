import random
import numpy as np
from typing import Iterator, List, Set
from bitmap_utils import create_funnel_pattern, create_random_grid, obstacle_at

_BLOCKED_COL = 0b111

class MapGenerator:
    def __init__(self, n_cols:int=7):
        assert n_cols > 0 and isinstance(n_cols, int), "n_cols must be a positive integer"
        # Dimensions
        self.ROWS = 3
        self.COLS = n_cols
        # Start in the middle lane
        start_col = n_cols // 2
        # State for generation: Tracks which columns the agent could validly be in
        self.valid_cols_prev: Set[int] = {start_col}
    
    def check_standing_survival(self, grid_bitmap:int, col:int):
        """Checks if you can survive a grid by standing in a given column"""
        stand_rows = (1, 2)
        for row in stand_rows:
            if obstacle_at(location=(row,col), grid_bitmap=grid_bitmap, map_width=self.COLS):
                return False
        return True

    def check_survival(self, grid_bitmap:int, col:int):
        """
        Checks if a specific column configuration allows ANY valid posture.
        Returns True if the agent can survive in this column by Jumping, Ducking, or Standing.
        """
        # column_slice is [row0_val, row1_val, row2_val]
        
        # Check Jump (Agent is at Row 0)
        can_jump = not obstacle_at(location=(0, col), grid_bitmap=grid_bitmap, map_width=self.COLS)
        
        # Check Stand (Agent is at Row 1 and 2)
        can_stand = self.check_standing_survival(grid_bitmap, col)
        
        # Check Duck (Agent is at Row 2)
        can_duck = not obstacle_at(location=(2, col), grid_bitmap=grid_bitmap, map_width=self.COLS)
        
        return can_jump or can_stand or can_duck

    def get_reachable_columns(self, current_col):
        """Returns list of columns reachable from current_col (left, stay, right)"""
        moves = [current_col - 1, current_col, current_col + 1]
        # Filter out moves that go off the edges of the N-lane map
        return [c for c in moves if 0 <= c < self.COLS]
    
    def reset(self):
        """Resets the generator state to the beginning (Time 0)."""
        # Reset to middle lane
        self.valid_cols_prev = {self.COLS // 2}

    def generate_step(self, trap_spawn_prob=0.5, difficulty = "medium") -> tuple[int, int]:
        """ 
        Generates single valid grid_bitmap and trap_col based on the previous state.
        trap_col is -1 if there is no trap.
        Parameters:
            trap_spawn_prob: probability (0 to 1) of a trap being generated given conditions are met
            difficulty: difficulty of the map
        Returns:
            (grid_bitmap, trap_col): the bitmap representation of the grid and the column of the trap 
                * trap_col is -1 if there is no trap
        """
        msg = "trap_spawn_prob must be between 0 and 1 inclusive"
        assert 0 <= trap_spawn_prob <= 1, msg
        if difficulty =="easy":
            obs_prob = 0.15
        elif difficulty == "medium":
            obs_prob = 0.30
        elif difficulty == "hard":
            obs_prob = 0.45
        else:  # expert
            obs_prob = 0.60

        valid_grid_found = False
        attempts = 0
        trap_col = -1
        final_grid_bitmap = 0

        while not valid_grid_found and attempts <100:
            candidate_grid_bitmap = 0
            # Dynamic Hard Mode Pattern (Works for any N width)
            # Creates a funnel by blocking all but one reachable column
            if difficulty in ("hard", "expert") and random.random() < 0.2:
                safe_prev_col = random.choice(list(self.valid_cols_prev))
                # normal case: randomly choose a safe col to the left or right
                safe_col = safe_prev_col + random.choice([-1, 1])
                # Handle edge cases
                if safe_col == -1:
                    safe_col = 1  # choose 1 rather than 0 so they still have to move from 0
                elif safe_col == self.COLS:
                    safe_col = self.COLS-2 # choose -2 so they still have to move from edge

                # block all but chosen col
                candidate_grid_bitmap = create_funnel_pattern(
                    safe_col=safe_col, n_rows=self.ROWS, map_width=self.COLS)
            else:
                # Random noise generation across all squares
                candidate_grid_bitmap = create_random_grid(
                    obstacle_frequency=obs_prob, n_rows=self.ROWS, map_width=self.COLS)

            # Solvability Check
            valid_cols_next = set()
            for prev_c in self.valid_cols_prev:
                possible_moves = self.get_reachable_columns(prev_c)
                # distinguish between current column and reachable columns (for survivability)
                curr_c = prev_c
                other_reachable_cols = [col for col in possible_moves if col != prev_c]
                # if any ducking/jumping/stayin in current column are survivable, add it
                if self.check_survival(grid_bitmap=candidate_grid_bitmap, col=curr_c):
                    valid_cols_next.add(curr_c)
                for move_c in other_reachable_cols:
                    if self.check_standing_survival(grid_bitmap=candidate_grid_bitmap, col=move_c):
                        valid_cols_next.add(move_c)

            if len(valid_cols_next) > 0:
                final_grid_bitmap = candidate_grid_bitmap
                self.valid_cols_prev = valid_cols_next
                valid_grid_found = True
                # Add in trap if there is more than one solution col
                # Does NOT guarantee a full map solution can avoid traps --> best policy can fail
                # if it did guarantee that, then traps could be replaced entirely with obstacles
                if trap_spawn_prob > 0 and len(valid_cols_next) > 1:
                    if random.random() < trap_spawn_prob:
                        trap_col = random.choice(list(valid_cols_next))

            else:
                attempts += 1

        if not valid_grid_found:
            # give a grid with no obstacles if no valid grid found
            final_grid_bitmap = 0

        return final_grid_bitmap, trap_col

    def infinite_track(self, trap_spawn_prob=0.5, difficulty="medium") -> Iterator[tuple[np.ndarray, int]]:
        """Yields 3xN grids and trap columns one by one FOREVER."""
        self.reset()
        # Whenever called, give a new slice
        while True:
            yield self.generate_step(trap_spawn_prob, difficulty)

    def generate_track(self, timesteps, trap_spawn_prob=0.5, difficulty="medium") -> tuple[List[np.ndarray], List[int]]:
        """Generates a fixed length track list and list of trap cols.
        Returns:
            (track, trap_cols): track and list of trap columns, both indexed at time t. A trap col of -1 indicates no trap at time t
        """
        self.reset()
        track = []
        trap_cols = []
        for _ in range(timesteps):
            grid, trap_col = self.generate_step(trap_spawn_prob, difficulty)
            track.append(grid)
            trap_cols.append(trap_col)
        return track, trap_cols

# --- Example Usage for Testing ---
def main():
    # Test with a wider map (e.g., 9 lanes)
    WIDTH_N = 9
    generator = MapGenerator(n_cols=WIDTH_N)
    print(f"Testing generate_step with {WIDTH_N} lanes:")
    generator.reset()
    for i in range(3):
        grid, trap_col = generator.generate_step(trap_spawn_prob=0.5, difficulty="expert")
        print(f"Slice {i}: \ntrap_col: {trap_col} \ngrid: \n{grid}")

if __name__ == "__main__":
    main()

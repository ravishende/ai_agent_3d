import random
import numpy as np
from typing import Iterator, List, Set

class MapGenerator:
    def __init__(self, n_cols=7):
        # Dimensions
        self.ROWS = 3
        self.COLS = n_cols
        # Start in the middle lane
        start_col = n_cols // 2
        # State for generation: Tracks which columns the agent could validly be in
        self.valid_cols_prev: Set[int] = {start_col}
    
    def check_standing_survival(self, column_slice):
        """Checks if you can survive by standing in a column"""
        return column_slice[1] == 0 and column_slice[2] == 0

    def check_survival(self, column_slice):
        """
        Checks if a specific column configuration allows ANY valid posture.
        Returns True if the agent can survive in this column by Jumping, Ducking, or Standing.
        """
        # column_slice is [row0_val, row1_val, row2_val]
        
        # Check Jump (Agent is at Row 0)
        can_jump = (column_slice[0] == 0)
        
        # Check Stand (Agent is at Row 1 and 2)
        can_stand = (column_slice[1] == 0 and column_slice[2] == 0)
        
        # Check Duck (Agent is at Row 2)
        can_duck = (column_slice[2] == 0)
        
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

    def generate_step(self, difficulty = "medium") -> np.ndarray:
        """ Generates exactly ONE single valid time-slice (3xN) based on the previous state """

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
        final_grid = np.zeros((self.ROWS,self.COLS), dtype=int)

        while not valid_grid_found and attempts <100:
            candidate_grid = np.zeros((self.ROWS,self.COLS), dtype=int)
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
                for col in range(self.COLS):
                    if col == safe_col:
                        continue
                    candidate_grid[:, col] = 1
            else:
                # Random noise generation across all N columns
                for r in range(self.ROWS):
                    for c in range(self.COLS):
                        if random.random() < obs_prob:
                            candidate_grid[r][c] = 1

            # Solvability Check
            valid_cols_next = set()
            for prev_c in self.valid_cols_prev:
                possible_moves = self.get_reachable_columns(prev_c)
                # distinguish between current column and reachable columns (for survivability)
                curr_c = prev_c
                other_reachable_cols = [col for col in possible_moves if col != prev_c]
                # if any ducking/jumping/stayin in current column are survivable, add it
                curr_col_slice = candidate_grid[:, curr_c]
                if self.check_survival(curr_col_slice):
                    valid_cols_next.add(curr_c)
                for move_c in other_reachable_cols:
                    # Numpy slicing to extract the column at move_c
                    col_slice = candidate_grid[:, move_c]
                    if self.check_standing_survival(col_slice):
                        valid_cols_next.add(move_c)

            if len(valid_cols_next) > 0:
                final_grid = candidate_grid
                self.valid_cols_prev = valid_cols_next
                valid_grid_found = True
            else:
                attempts += 1

        if not valid_grid_found:
            final_grid = np.zeros((self.ROWS, self.COLS), dtype=int)

        return final_grid

    def infinite_track(self, difficulty="medium") -> Iterator[np.ndarray]:
        """Yields 3xN grids one by one FOREVER."""
        self.reset()
        # Whenever called, give a new slice
        while True:
            yield self.generate_step(difficulty)

    def generate_track(self, timesteps, difficulty="medium") -> List[np.ndarray]:
        """Generates a fixed length track list."""
        self.reset()
        track = []
        for _ in range(timesteps - 1):
            track.append(self.generate_step(difficulty))
        return track

# --- Example Usage for Testing ---
if __name__ == "__main__":
    # Test with a wider map (e.g., 9 lanes)
    WIDTH_N = 9
    generator = MapGenerator(n_cols=WIDTH_N)
    print(f"Testing generate_step with {WIDTH_N} lanes:")
    generator.reset()
    for i in range(3):
        print(f"Slice {i}: \n{generator.generate_step(difficulty="expert")}")

import random
import numpy as np

class MapGenerator:
    def __init__(self):
        # Dimensions
        self.ROWS = 3
        self.COLS = 3
        
    def check_standing_survival(self, column_slice):
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
        return [c for c in moves if 0 <= c < self.COLS]

    def generate_track(self, timesteps, difficulty="medium"):
        """
        Generates a solvable track.
        
        Args:
            timesteps (int): Total length of track.
            difficulty (str): 'easy', 'medium', 'hard'
        
        Returns:
            list[np.ndarray]: List of 3x3 grids compliant with core.py
        """
        
        # Difficulty settings (Probability of an obstacle appearing in a cell)
        if difficulty == "easy":
            obs_prob = 0.15
        elif difficulty == "medium":
            obs_prob = 0.30
        elif difficulty == "hard":
            obs_prob = 0.45
        else:  # expert
            obs_prob = 0.60

        track = []
        
        # Step 0 is always empty to give the agent a fair start
        # MUST BE NP.ARRAY to match core.py expectations
        empty_grid = np.zeros((self.ROWS, self.COLS), dtype=int)
        track.append(empty_grid)
        
        valid_cols_prev = {1} 

        for t in range(1, timesteps):
            valid_grid_found = False
            attempts = 0
            
            while not valid_grid_found and attempts < 100:
                # 1. Generate a candidate grid (Temporary Python list for easy editing)
                candidate_grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
                
                # For 'hard', sometimes force a specific pattern (funnel)
                if difficulty in ['hard', 'expert'] and random.random() < 0.2:
                    blocked_col = random.choice([0, 2])
                    for r in range(self.ROWS):
                        candidate_grid[r][blocked_col] = 1
                        candidate_grid[r][1] = 1
                else:
                    for r in range(self.ROWS):
                        for c in range(self.COLS):
                            if random.random() < obs_prob:
                                candidate_grid[r][c] = 1

                # 2. Solvability Check
                valid_cols_next = set()
                
                for prev_c in valid_cols_prev:
                    possible_moves = self.get_reachable_columns(prev_c)
                    curr_c = prev_c
                    # distinguish between current column and reachable columns
                    other_reachable_cols = [col for col in possible_moves if col != prev_c]
                    curr_col_slice = [candidate_grid[i][curr_c] for i in range(self.ROWS)]
                    # if any ducking/jumping/stayin in current column are survivable, add it
                    if self.check_survival(curr_col_slice):
                        valid_cols_next.add(curr_c)
                    # if standing in a reachable column is survivable, add it.
                    # (can't move laterally and jump in one turn)
                    for move_c in other_reachable_cols:
                        col_slice = [candidate_grid[i][move_c] for i in range(self.ROWS)]
                        if self.check_standing_survival(col_slice):
                            valid_cols_next.add(move_c)

                # 3. Finalize
                if len(valid_cols_next) > 0:
                    # CONVERT TO NUMPY ARRAY BEFORE APPENDING
                    track.append(np.array(candidate_grid))
                    valid_cols_prev = valid_cols_next
                    valid_grid_found = True
                else:
                    attempts += 1
            
            if not valid_grid_found:
                track.append(empty_grid.copy())
        
        return track

# --- Example Usage for Testing ---
def main():
    generator = MapGenerator()
    track = generator.generate_track(10, "medium")
    print(f"Generated {len(track)} grids.")
    print(f"Type of grid: {type(track[0])}") # Should be <class 'numpy.ndarray'>

if __name__ == "__main__":
    main()
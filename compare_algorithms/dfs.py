from typing import List, Optional, Set, Tuple
from core import State, Action, GET_START_LOCATION

DFS_ACTIONS = [
    Action.LEFT,
    Action.RIGHT,
    Action.JUMP,
    Action.DUCK,
    Action.STAY,
]

def _dfs_step_allow_traps(state: State, action: Action) -> Optional[State]:
    if state.is_terminal():
        return state
    new_row, new_col = state._update_location(state.player_col, action)
    if state._collides((new_row, new_col), state.grid_bitmap):
        return None
    return State(state.time_index + 1, new_col)

def _dfs_step_avoid_traps(state: State, action: Action) -> Optional[State]:
    if state.is_terminal():
        return state
    new_row, new_col = state._update_location(state.player_col, action)
    if state._collides((new_row, new_col), state.grid_bitmap):
        return None
    if state.trap_col != -1 and new_col == state.trap_col:
        return None
    return State(state.time_index + 1, new_col)

def dfs_find_safe_path(start_state, dfs_step_func, max_depth: int | None = None) -> List[Action]:
    visited: Set[Tuple[int, int]] = set()
    best_complete: Optional[List[Action]] = None
    best_partial: List[Action] = []

    # Stack holds tuples: (state, iterator index for DFS_ACTIONS, current path list)
    stack: List[Tuple[State, int, List[Action]]] = []
    stack.append((start_state, 0, []))

    while stack:
        state, next_action_idx, path = stack.pop()

        # reached goal
        if state.is_terminal():
            if best_complete is None or len(path) < len(best_complete):
                best_complete = path
            continue
        if max_depth is not None and len(path) >= max_depth:
            if len(path) > len(best_partial):
                best_partial = path
            continue

        key = (state.time_index, state.player_col)
        if key in visited:
            continue
        visited.add(key)
        any_move = False
        # Iterate through all actions from DFS_ACTIONS
        for action in DFS_ACTIONS:
            next_state = dfs_step_func(state, action)
            if next_state is None:
                continue
            any_move = True
            stack.append((next_state, 0, path + [action]))
        if not any_move and len(path) > len(best_partial):
            best_partial = path

    if best_complete is not None:
        return best_complete
    if best_partial:
        return best_partial
    return []

def dfs_allow_traps(game_map, trap_cols):
    start_state = State(0, game_map[0].shape[1]//2)
    return dfs_find_safe_path(start_state, _dfs_step_avoid_traps)

def dfs_avoid_traps(game_map, trap_cols):
    start_state = State(0, game_map[0].shape[1]//2)
    return dfs_find_safe_path(start_state, _dfs_step_allow_traps)
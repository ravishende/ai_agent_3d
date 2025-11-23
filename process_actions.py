from core import START_LOCATION, Action




def main():
    action_strs = ['r', 'j', 'd', 'l', 's', 'd', 's', 'r']
    locations = get_locations_list(action_strs)
    print(locations)

def get_locations_list(actions_list:list[str] | list[Action]) -> list[tuple[int, int]]:
    """Given a list of actions (as strings or actions), return a list of locations of the player during each action"""
    actions: list[Action] = _process_actions(actions_list)
    curr_location = START_LOCATION
    locations = []
    for action in actions:
        during_loc, after_loc = _get_locations(curr_location, action)
        locations.append(during_loc)
        curr_location = after_loc
    return locations



def _process_actions(actions: list[str] | list[Action]) -> list[Action]:
    processed_actions = []
    for action in actions:
        if isinstance(action, Action):
            processed_actions.append(action)
        elif isinstance(action, str):
            processed_actions.append(_str_to_action(action))
        else:
            raise TypeError(f"action {action} must be a str or Action but was {type(action)}")
    return processed_actions

def _str_to_action(move:str):
    "Given a move (l, r, j, d, s), return an Action"
    actions = {
        "l": Action.LEFT,
        "r": Action.RIGHT,
        "j": Action.JUMP,
        "d": Action.DUCK,
        "s": Action.STAY
    }
    move = move.lower()
    if move not in actions:
        raise KeyError(f"Chosen move not in accepted moves: {list(actions.keys())}")
    return actions[move]

def _get_locations(curr_location: tuple[int, int],
                    action:Action) -> tuple[tuple[int, int], tuple[int, int]]:
    """Given the current location and action, return the location during and immediately after the slice.
    Used for getting locations to rendering graphics.
    
    In other words, return the player location while dodging obstacles and the reset position after.
    Ex: update_location((1,1), Action.JUMP) would return (1,0), (1,1) because they are jumping -->> (1,0) during the slice and then standing --> (1,1) immediately after the slice.
    
    If chaining together get_locations() calls, the second location (after slice) is the one to
    use as the curr_location parameter for the next call.

    Parameters:
        curr_location: location before the action (row,col)
        action: action to be taken
    Returns:
        location_during, location_after
    """
    row, col = curr_location
    row_change, col_change = action.value
    # update the row and column, making sure they stay in bounds of 0 and 2
    new_row = max(0, row + row_change)
    new_row = min(new_row, 2)
    new_col = max(0, col + col_change)
    new_col = min(new_col, 2)

    standing_row = 1
    location_during = (new_row, new_col)
    location_after = (standing_row, new_col)
    return location_during, location_after


if __name__ == "__main__":
    main()
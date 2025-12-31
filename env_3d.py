"""
Visualizes a map and player.
Player lives at X=0, fixed relative to camera, while map moves towards them (decreasing x)
Player can move in the Y and Z axis for actions of JUMP/DUCK and LEFT/RIGHT respectivelly
The view is of a YZ plane, with X extending into the distance. (Y is up, Z is right)
"""
import pygame
import random
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from maps import maps
from core import INIT_START_LOCATION, GET_START_LOCATION

# --------------------- Colors ----------------------
GREEN = (0.2, 0.8, 0.3)
RED = (0.9, 0.2, 0.2)
BLUE = (0.2, 0.2, 0.9)
BLACK = (0.0, 0.0, 0.0)

# ------------------ Cube geometry ------------------

# vertices for centered 1-unit cube
VERTICES = [
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5]
]

# faces defined as lists of vertex indices
FACES = [
    (0, 1, 2, 3),  # back
    (4, 5, 6, 7),  # front
    (0, 1, 5, 4),  # bottom
    (2, 3, 7, 6),  # top
    (1, 2, 6, 5),  # right
    (0, 3, 7, 4)   # left
]

# edges for outline of cubes
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def _draw_cube_outline(size=1.0):
    """Draw just the black outline of a cube with given edge length."""
    glColor3f(*BLACK)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for e in EDGES:
        for idx in e:
            vx, vy, vz = VERTICES[idx]
            glVertex3f(vx * size, vy * size, vz * size)
    glEnd()

def _draw_cube(color, size=1.0):
    """Draw a solid-colored cube with no outline."""
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glColor3f(*color)
    glBegin(GL_QUADS)
    for face in FACES:
        for idx in face:
            vx, vy, vz = VERTICES[idx]
            glVertex3f(vx * size, vy * size, vz * size)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)

def _draw_colored_cube_with_outline(color, size=1.0):
    """Draw a solid-colored cube with a black outline."""
    solid_size = 0.98*size  # smaller than outline to avoid z-fighting
    outline_size = size
    # solid faces
    _draw_cube(color, solid_size)
    # outline - slightly larger to avoid z-fighting
    _draw_cube_outline(outline_size)

def _grid_to_world(row_index, col_index, cube_size, num_rows=3):
    """
    Logical grid coord -> world-space (y, z).
    Player coordinate (1,2) means:
        row_index = 1 --> 2nd row from top
        col_index = 2 --> 3rd row from left
    Returns:
        y_position, z_position: resulting y and z position of cube
    """
    y = (num_rows - 1 - row_index) * cube_size
    z = col_index * cube_size
    return y, z

# ------------------ Floor rendering ------------------

def _draw_floor(map_length, map_width=3, cube_size=1.0, spacing=1.0, color=(0.5,0.5,0.5)):
    """Draw a floor of the map for the players and obstacles to rest on
    
    Parameters:
        map_length: number of slices in the map
        map_width: number of columns in a slice (for a 3xn slice, it would be n)
        cube_size: the sidelength of a cube in a slice
        spacing: how far apart slices are
        color (tuple of 3 floats from 0 to 1): color of the floor
    Returns:
        None
    """
    step = cube_size+spacing
    padding = 1 * step # how much extra to add to width and length of floor
    floor_height = 0.25
    floor_length = step * map_length + 2*padding
    floor_width = cube_size * map_width + padding

    half_height = floor_height/2
    half_length = floor_length/2
    half_width = floor_width/2
    floor_vertices = [
        # x is length, y is height, z is width
        [-half_length, -half_height, -half_width],
        [-half_length, -half_height,  half_width],
        [-half_length,  half_height, -half_width],
        [-half_length,  half_height,  half_width],
        [ half_length, -half_height, -half_width],
        [ half_length, -half_height,  half_width],
        [ half_length,  half_height, -half_width],
        [ half_length,  half_height,  half_width],
    ]
    x_offset = floor_length/2 - _get_lane_x_start(cube_size, spacing)
    # bottom row of cubes are centered at y=0. Be below that
    y_offset = -(cube_size/2 + 0.2)
    # Let caller handle Z-centering; we stay centered around z = 0 here
    z_offset = 0.0

    glPushMatrix()
    glTranslatef(x_offset, y_offset, z_offset)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glColor3f(*color)
    glBegin(GL_QUADS)
    for face in FACES:
        for idx in face:
            vx, vy, vz = floor_vertices[idx]
            glVertex3f(vx, vy, vz)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)
    glPopMatrix()

# ------------------ Lane rendering ------------------

def _draw_lane_windowed(
    slices:list[np.ndarray],
    trap_cols:list[int] | None,
    lane_x_position_world,  # lane_x_position (the value to translate the full lane by)
    cube_size:float = 2.0,
    spacing:float = 40.0,
    num_rows:int = 3,
    pad_slices:int = 2,  # extra slices behind/ahead to avoid pop-in
    x_near:float = -20.0,
    x_far:float =200.0,
    kill_time_column: tuple[int, int] | None = None, # (slice_index, col)
):
    """
    Draw only the visible portion of the lane.

    Draw only the slices whose WORLD-X lies in [x_near, x_far] (plus pad_slices).
    Assumes caller has not already translated by lane_x_position (translattion happens inside),

    Parameters:
        slices: game map as a list of 2D arrays
        trap_cols: trap column indices for each slice. -1 means no trap in that slice.
        lane_x_position_world: world-space X position of slice index 0. 
            * world_x(i) = lane_x_position_world + i * (cube_size + spacing)
        cube_size: side length of each cube in world units.
        spacing: distance between consecutive slices along the X axis.
        num_rows: number of rows in each slice (Y dimension).
        pad_slices: number of extra slices to render before/after the visible window to avoid visual popping as slices enter or leave the view.
        x_near: world-space X coordinate of the near edge of the visible window.
        x_far: world-space X coordinate of the far edge of the visible window.
    """
    if not slices:
        return

    map_len = len(slices)
    _, cols = slices[0].shape
    step = cube_size + spacing

    # Center the lane in Z so the middle column is at z = 0
    center_z_offset = ((cols - 1) * cube_size) / 2

    # Compute visible slice index range based on world_x(i) = lane_x_position + i*step
    i_start = int(np.floor((x_near - lane_x_position_world) / step)) - pad_slices
    i_end   = int(np.ceil ((x_far  - lane_x_position_world) / step)) + pad_slices
    i_start = max(i_start, 0)
    i_end   = min(i_end, map_len - 1)

    if i_end < i_start:
        return

    visible_count = i_end - i_start + 1

    # draw a short floor just for visible slices
    floor_color = (0.5, 0.5, 0.5)
    glPushMatrix()
    # move to lane, then move floor start to i_start
    glTranslatef(lane_x_position_world + i_start * step, 0.0, 0.0)
    _draw_floor(
        map_length=visible_count,
        map_width=cols,
        cube_size=cube_size,
        spacing=spacing,
        color=floor_color
    )
    glPopMatrix()

    # draw only visible slices
    glPushMatrix()
    glTranslatef(lane_x_position_world, 0.0, 0.0)      # lane motion
    glTranslatef(0.0, 0.0, -center_z_offset)           # Z centering

    for i in range(i_start, i_end + 1):
        x = i * step
        grid = slices[i]

        # cubes
        for row in range(num_rows):
            for col in range(cols):
                if grid[row, col] == 1:
                    y = (num_rows - 1 - row) * cube_size
                    z = col * cube_size

                    if row != num_rows - 1:
                        glPushMatrix()
                        glTranslatef(x, 0.0, z)
                        _draw_shadow(cube_size)
                        glPopMatrix()

                    glPushMatrix()
                    glTranslatef(x, y, z)
                    _draw_colored_cube_with_outline(GREEN, cube_size)
                    glPopMatrix()

        # traps
        if trap_cols is not None and trap_cols[i] != -1:
            z = trap_cols[i] * cube_size
            trap_display_length = 1
            if spacing > trap_display_length * cube_size:
                trap_display_length = 2
            for k in range(trap_display_length + 1):
                glPushMatrix()
                glTranslatef(x - cube_size * k, 0.0, z)
                _draw_trap(cube_size)
                glPopMatrix()

        # kill column (vertical red blocks) if the player died to a trap on this slice
        if kill_time_column is not None:
            kill_slice, kill_col = kill_time_column
            if i == kill_slice:
                z = kill_col * cube_size
                glPushMatrix()
                glTranslatef(x, 0.0, z)
                _draw_trap_kill_column(cube_size=cube_size, num_rows=num_rows)
                glPopMatrix()

    glPopMatrix()

# ------------------ OpenGL / Pygame setup ------------------

def _init_pygame_opengl(width=800, height=600, cube_size=1.0):
    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.05, 1.0)
    glDisable(GL_LIGHTING)

    glEnable(GL_MULTISAMPLE)

    fov = 40.0
    aspect = width / float(height)
    gluPerspective(fov, aspect, 0.1, 200.0)
    # move the world away from the camera (equivalent to moving camera up and back)
    glTranslatef(0.0, -2*cube_size, -30.0)
    # rotate the world to the correct orientation
    glRotatef(10, 1, 0, 0)  # have the camera look down on the world
    glRotatef(90, 0, 1, 0)

# ------------------ Distance retrieval ------------------

def _get_lane_x_start(cube_size, spacing, steps_back=1):
    """Get the x position for the lane to start at
    Parameters:
        cube_size: sidelength of a cube in a slice
        spacing: how far apart slices are
        steps_back: how many steps back to push the start
    Returns:
        x_distance: the x position for the lane to start at
    """
    step = cube_size + spacing
    return spacing + step * steps_back

# ------------------ Shadows and traps ------------------

def _draw_square(cube_size:float, color:tuple[float, float, float]):
    """
    Draw a simple flat square on the floor with a given color
    Assumes already in correct XZ-coordinates, with y=0 at the center of the bottom row cubes.
    """
    # Slightly below the bottom row cube center, just above the floor
    shadow_y = -(cube_size / 2.0 + 0.05)
    half_x = cube_size * 0.49
    half_z = cube_size * 0.49

    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-2.0, -2.0)  # push it in front of the floor to avoid z-fighting
    glColor3f(*color)

    glBegin(GL_QUADS)
    glVertex3f(-half_x, shadow_y, -half_z)
    glVertex3f(-half_x, shadow_y,  half_z)
    glVertex3f( half_x, shadow_y,  half_z)
    glVertex3f( half_x, shadow_y, -half_z)
    glEnd()

    glDisable(GL_POLYGON_OFFSET_FILL)

def _draw_shadow(cube_size):
    """
    Draw a simple flat shadow square on the floor (for beneath floating cube).
    Assumes already in correct XZ-coordinates, with y=0 at the center of the bottom row cubes.
    """
    _draw_square(cube_size, BLACK)

def _draw_trap(cube_size):
    """
    Draw a simple flat trap square on the floor.
    Assumes already in correct XZ-coordinates, with y=0 at the center of the bottom row cubes.
    """
    _draw_square(cube_size, RED)

# ------------------ Death logic ------------------

def _player_occupied_cells(player_pos: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a list of player positions (row,col) that the player occupies"""
    row, col = player_pos
    stand_row = 1
    bot_row = 2
    if row == stand_row:
        # player occupies 2 spaces when standing
        return [(stand_row, col), (bot_row, col)]
    # player occupies 1 space when jumping/ducking
    return [(row, col)]

def _hits_obstacle(slice_grid: np.ndarray, player_pos: tuple[int, int]) -> bool:
    """True if any occupied player cell is a 1 in this slice."""
    for row, col in _player_occupied_cells(player_pos):
        if slice_grid[row, col] == 1:
            return True
    return False


def _hits_trap(trap_cols: list[int] | None, slice_index: int, player_pos: tuple[int, int]) -> bool:
    """
    Trap kills player if they are in same column as trap during the slice the trap is in.
    """
    if trap_cols is None:
        return False
    trap_col = trap_cols[slice_index]
    if trap_col == -1:
        return False
    _, player_col = player_pos
    return player_col == trap_col

def _draw_trap_kill_column(cube_size: float, num_rows: int):
    """
    Draw a vertical red column (stack of cubes) from floor up through all rows.
    Caller must already be translated to the correct (x, z) for the column.
    """
    for row in range(num_rows):
        y = (num_rows - 1 - row) * cube_size
        glPushMatrix()
        glTranslatef(0.0, y, 0.0)
        _draw_colored_cube_with_outline(RED, cube_size)
        glPopMatrix()

# ------------------ Main loop ------------------

def visualize(
        game_map:list[np.ndarray],
        player_locations:list[tuple[int,int]],
        trap_cols:list[int] | None = None,
        cube_size: float = 2.0,
        spacing: float = 40.0):
    """Given a game map and during-action payer locations, visualize a character moving through the map"""
    slices = game_map  # game map
    cols = slices[0].shape[1]
    INIT_START_LOCATION(map_width=cols)

    # Compute lane centering offset in Z (used to center the player)
    center_z_offset = ((cols - 1) * cube_size) / 2.0

    _init_pygame_opengl(
        width=800,
        height=600,
        cube_size=cube_size)

    running = True
    clock = pygame.time.Clock()

    # lane dimension and position
    step = cube_size + spacing
    # how far the lane starts toward the camera (in +X)
    lane_x_start = _get_lane_x_start(cube_size=cube_size, spacing=spacing)
    lane_length = len(slices) * step
    lane_x_position = lane_x_start
    # movement speed per frame
    speed = 25
    # when lane passes this, jump it back to beginning
    reset_distance = 0
    # how far into a section to change to the newest action
    action_change = 1/2
    action_change_distance = action_change * step

     # death / game over handling
    death_active = False
    death_timer = 0.0
    DEATH_SHOW_SECONDS = 0.9  # how long to pause and show the kill column
    death_kill_column: tuple[int, int] | None = None  # (slice_index, col)

    while running:
        fps = 120
        dt_ms = clock.tick(fps) # ms since last frame
        dt = dt_ms / 1000.0
        fps = clock.get_fps()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # If showing a death, freeze motion and count down, then reset.
        if death_active:
            death_timer -= dt
            if death_timer <= 0.0:
                death_active = False
                death_kill_column = None
                lane_x_position = lane_x_start
            # (We still draw the scene below, with the kill column visible.)
        else:
            # move lane toward camera
            lane_x_position -= speed * dt
            if lane_x_position + lane_length < reset_distance:
                lane_x_position = lane_x_start  # restart from back

        total_distance_travelled = lane_x_start - lane_x_position
        map_distance_travelled = total_distance_travelled - lane_x_start + spacing
        current_slice = int((map_distance_travelled - action_change_distance) // step)
        # don't go out of range of slices
        current_slice = max(current_slice, 0)
        current_slice = min(current_slice, len(slices)-1)
        # section progress is how far into a section (right after prev blocks to end of next blocks) you've gone.
        # section_progress values are btwn (1 - action_change) and action_change (e.g. -1/3, 2/3)
        # Ideally it'd be between 0 and 1, but only needs to be accurate between 1/4 and action_change
        section_progress = (map_distance_travelled - step*(current_slice+1)) / step

        in_action_phase = False
        if map_distance_travelled < action_change_distance:
            # start the game standing at (cols//2,1) - in the middle of the board
            player_pos = GET_START_LOCATION()
        elif 1/4 <= section_progress < action_change_distance/step:
            # reset after previous action: return to standing in the same col as the prev action
            _, prev_col = player_locations[current_slice]
            stand_row = 1
            player_pos = (stand_row, prev_col)
        else:
            # do the action for the current slice
            player_pos = player_locations[current_slice]
            in_action_phase = True

        # Only check collisions during action phase for the current slice.
        death_phase_stop_threshold = -0.075
        if (not death_active) and in_action_phase and death_phase_stop_threshold <= section_progress:
            # obstacle collision (if you want last-slice obstacle deaths to show too)
            if _hits_obstacle(slices[current_slice], player_pos):
                death_active = True
                death_timer = DEATH_SHOW_SECONDS
                death_kill_column = None  # obstacle death: no trap column
            # trap collision
            elif _hits_trap(trap_cols, current_slice, player_pos) or _hits_obstacle(slices[current_slice], player_pos):
                _, player_col = player_pos
                death_active = True
                death_timer = DEATH_SHOW_SECONDS
                death_kill_column = (current_slice, player_col)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ---- Draw player ----
        glPushMatrix()
        player_color = BLUE
        # first cube: top half/torso of player
        player_y, player_z = _grid_to_world(player_pos[0], player_pos[1], cube_size)
        # shift in Z so middle column is around z = 0
        glTranslatef(0.0, player_y, player_z - center_z_offset)
        _draw_colored_cube_with_outline(player_color, cube_size)

        # Second cube (bottom half of player) depends on posture:
        # row == 1: standing vertically (feet below the torso)
        player_row = player_pos[0]
        if player_row == 1:
            glTranslatef(0.0, -cube_size, 0.0)
        # row == 0: diving horizontally (feet behind the torso)
        elif player_row == 0:
            glTranslatef(cube_size, 0.0, 0.0)
        # row == 2: sliding horizontally (feet in front of the torso)
        else:
            glTranslatef(-cube_size, 0.0, 0.0)

        _draw_colored_cube_with_outline(player_color, cube_size)
        glPopMatrix()

        # ---- Draw lane ----
        _draw_lane_windowed(
            slices, trap_cols,
            lane_x_position_world=lane_x_position,
            cube_size=cube_size,
            spacing=spacing,
            x_near=-20.0,
            x_far=200.0,
            pad_slices=2,
            kill_time_column=death_kill_column
        )

        pygame.display.flip()

    pygame.quit()

def main():
    game_map = maps[1]
    locations = [(1, 2), (0, 2), (2, 2), (1, 1), (1, 1), (2, 1), (1, 1), (1, 2)]
    trap_cols = [random.randint(0,2) for _ in range(len(game_map))]
    cube_size = 2.0
    spacing = 40.0  # how far apart slices are
    visualize(game_map=game_map,
              player_locations=locations,
              trap_cols=trap_cols,
              cube_size=cube_size,
              spacing=spacing)

if __name__ == "__main__":
    main()

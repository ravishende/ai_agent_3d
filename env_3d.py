"""
Visualizes a map and player.
Player lives at X=0, fixed relative to camera, while map moves towards them (decreasing x)
Player can move in the Y and Z axis for actions of JUMP/DUCK and LEFT/RIGHT respectivelly
"""

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from maps import maps
from core import START_LOCATION

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
    glColor3f(0.0, 0.0, 0.0)  # black
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
    # TODO: find vertices of an object given the height, width, and length
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
    # col 0 of slice is centered at z=0 --> shift floor so it's centered with right-shifted slice
    z_offset = cube_size * ((map_width-1) // 2)
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

def _draw_lane_from_slices(slices, cube_size=1.0, spacing=1.0, num_rows=3):
    """
    slices: list of 3xN numpy arrays with 0/1 values.
    Each array is a cross-section in the Y-Z plane.
    The list index is the X position along the lane.
    
    Given a 3xN array, array[0][0] is placed so its cube center is at y=0, z=0
    """
    if not slices:
        return

    rows, cols = slices[0].shape

    floor_color = (0.5, 0.5, 0.5)
    _draw_floor(
        map_length=len(slices),
        map_width=rows,
        cube_size=cube_size,
        spacing=spacing,
        color=floor_color)

    step = cube_size + spacing
    x_offset = 0.0

    for i, grid in enumerate(slices):
        x = x_offset + i * step

        for row in range(rows):
            for col in range(cols):
                if grid[row, col] == 1:
                    y = (num_rows - 1 - row) * cube_size
                    z = col * cube_size

                    glPushMatrix()
                    glTranslatef(x, y, z)
                    color = (0.2, 0.8, 0.3)  # mono green-ish
                    _draw_colored_cube_with_outline(color, cube_size)
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
    """
    step = cube_size + spacing
    return spacing + step * steps_back

# ------------------ Main loop ------------------

def visualize(game_map:list[np.ndarray], player_locations:list[tuple[int,int]]):
    """Given a game map and during-action payer locations, visualize a character moving through the map"""
    slices = game_map  # game map
    cube_size = 2.0
    spacing = 40.0  # how far apart slices are

    _init_pygame_opengl(
        width=800,
        height=600,
        cube_size=cube_size)

    running = True
    clock = pygame.time.Clock()

    # lane dimension and position
    step = cube_size + spacing
    # how far the lane starts toward the camera (in +X)
    lane_x_start = spacing + 1*step
    lane_length = len(slices) * step
    lane_x_position = lane_x_start
    # movement speed per frame
    speed = 25
    # when lane passes this, jump it back to beginning
    reset_distance = 0
    # how far into a section to change to the newest action
    action_change = 1/2
    action_change_distance = action_change * step
    while running:
        fps = 120
        dt_ms = clock.tick(fps) # ms since last frame
        dt = dt_ms / 1000.0
        fps = clock.get_fps()


        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

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

        if len(player_locations) < len(slices):  # player didn't win - lost early
            if current_slice == len(player_locations) -1 and section_progress >= 0:
                # player hit an obstacle --> reset
                lane_x_position = lane_x_start
                continue

        if map_distance_travelled < action_change_distance:
            # start the game standing at (1,1) - in the middle of the 3x3 board
            player_pos = START_LOCATION
        elif 1/4 <= section_progress < action_change_distance/step:
            # reset after previous action: return to standing in the same col as the prev action
            _, prev_col = player_locations[current_slice]
            stand_row = 1
            player_pos = (stand_row, prev_col)
        else:
            # do the action for the current slice
            player_pos = player_locations[current_slice]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glPushMatrix()
        player_color = (0.9, 0.2, 0.2)  # reddish color
        # first cube: top half/torso of player
        player_y, player_z = _grid_to_world(player_pos[0], player_pos[1], cube_size)
        glTranslatef(0.0, player_y, player_z)
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

        glPushMatrix()
        # move entire lane along -X (toward camera)
        glTranslatef(lane_x_position, 0.0, 0.0)
        _draw_lane_from_slices(slices, cube_size, spacing)
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()

def main():
    locations = [(1, 2), (0, 2), (2, 2), (1, 1), (1, 1), (2, 1), (1, 1), (1, 2)]
    visualize(maps[1], locations)

if __name__ == "__main__":
    main()    

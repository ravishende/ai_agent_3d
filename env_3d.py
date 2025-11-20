"""
Visualizes a map and player.
Player lives at X=0, fixed relative to camera, while map moves towards them (decreasing x)
Player can move in the Y and Z axis for actions of LEFT/RIGHT and JUMP/DUCK
"""


import math
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from maps import map2

# ------------------ Cube geometry ------------------

# vertices for centered 1-unit cube
vertices = [
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
faces = [
    (0, 1, 2, 3),  # back
    (4, 5, 6, 7),  # front
    (0, 1, 5, 4),  # bottom
    (2, 3, 7, 6),  # top
    (1, 2, 6, 5),  # right
    (0, 3, 7, 4)   # left
]

# edges for outline of cubes
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def draw_cube_outline(size=1.0):
    """Draw just the black outline of a cube with given edge length."""
    glColor3f(0.0, 0.0, 0.0)  # black
    glLineWidth(2.0)

    glBegin(GL_LINES)
    for e in edges:
        for idx in e:
            vx, vy, vz = vertices[idx]
            glVertex3f(vx * size, vy * size, vz * size)
    glEnd()

def draw_cube(color, size=1.0):
    """Draw a solid-colored cube with no outline."""
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)
    glColor3f(*color)
    glBegin(GL_QUADS)
    for face in faces:
        for idx in face:
            vx, vy, vz = vertices[idx]
            glVertex3f(vx * size, vy * size, vz * size)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)

def draw_colored_cube_with_outline(color, size=1.0):
    """Draw a solid-colored cube with a black outline."""
    solid_size = 0.98*size  # smaller than outline to avoid z-fighting
    outline_size = size
    # solid faces
    draw_cube(color, solid_size)
    # outline - slightly larger to avoid z-fighting
    draw_cube_outline(outline_size)


# ------------------ Lane rendering ------------------

def draw_lane_from_slices(slices, cube_size=1.0, spacing=1.0):
    """
    slices: list of 3x3 numpy arrays with 0/1 values.
    Each array is a cross-section in the Y-Z plane.
    The list index is the X position along the lane.
    """
    if not slices:
        return

    rows, cols = slices[0].shape

    step = cube_size + spacing
    x_offset = 0.0
    y_offset = - (rows - 1) * cube_size / 2.0
    z_offset = - (cols - 1) * cube_size / 2.0

    for i, grid in enumerate(slices):
        x = x_offset + i * step

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    y = y_offset + (rows - 1 - r) * cube_size
                    z = z_offset + c * cube_size

                    glPushMatrix()
                    glTranslatef(x, y, z)

                    color = (0.2, 0.8, 0.3)  # mono green-ish
                    draw_colored_cube_with_outline(color, cube_size)
                    glPopMatrix()

# ------------------ OpenGL / Pygame setup ------------------

def init_pygame_opengl(num_slices, width=800, height=600, cube_size=1.0):
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


# ------------------ Main loop ------------------

def main():
    slices = map2  # game map
    cube_size = 2.0
    spacing = 40.0  # how far apart slices are

    init_pygame_opengl(
        num_slices=len(slices),
        width=800,
        height=600,
        cube_size=cube_size)

    running = True
    clock = pygame.time.Clock()

    # how far the lane starts toward the camera (in +X)
    lane_x_start = cube_size + spacing
    # lane dimension and position
    lane_length = len(slices) * (cube_size + spacing)
    lane_x_position = lane_x_start
    # movement speed per frame
    speed = 25
    # when lane passes this, jump it back to beginning
    reset_distance = 0

    while running:
        fps = 120
        dt_ms = clock.tick(fps) # ms since last frame
        dt = dt_ms / 1000.0
        fps = clock.get_fps()
        print(f"FPS: {fps:.1f}", end='\r')


        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # move lane toward camera
        lane_x_position -= speed * dt
        if lane_x_position + lane_length < reset_distance:
            lane_x_position = lane_x_start  # restart from back

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        glPushMatrix()
        player_color = (0.9, 0.2, 0.2)  # reddish color
        # top cube: top half of player
        draw_colored_cube_with_outline(player_color, cube_size)
        # bottom cube: bottom half of player
        glTranslatef(0.0, -cube_size, 0.0)
        draw_colored_cube_with_outline(player_color, cube_size)
        glPopMatrix()

        glPushMatrix()
        # move entire lane along -X (toward camera)
        glTranslatef(lane_x_position, 0.0, 0.0)
        draw_lane_from_slices(slices, cube_size, spacing)
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

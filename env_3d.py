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

def draw_colored_cube_with_outline(color, size=1.0):
    """Draw a solid-colored cube with a black outline."""
    # solid faces
    glColor3f(*color)
    glBegin(GL_QUADS)
    for face in faces:
        for idx in face:
            vx, vy, vz = vertices[idx]
            glVertex3f(vx * size, vy * size, vz * size)
    glEnd()

    # outline - slightly larger to avoid z-fighting
    outline_size = size * 1.01
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

    num_slices = len(slices)
    rows, cols = slices[0].shape

    step = cube_size + spacing
    x_offset = - (num_slices - 1) * step / 2.0
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

def init_pygame_opengl(num_slices, width=800, height=600, cube_size=1.0, spacing=1.0):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.05, 1.0)
    glDisable(GL_LIGHTING)

    fov = 60.0
    aspect = width / float(height)
    gluPerspective(fov, aspect, 0.1, 2000.0)

    # --- compute a reasonable camera distance based on cube size & lane length ---
    lane_length = (num_slices - 1) * (cube_size + spacing) + cube_size
    lane_height = 3 * cube_size  # since your cross-sections are 3 cubes tall

    # max half-extent of the object (rough bounding sphere radius)
    half_diag = 0.5 * math.sqrt(lane_length**2 + lane_height**2)
    # distance so that this half-diagonal fits into the vertical FOV
    distance = half_diag / math.tan(math.radians(fov / 2.0))
    # add a bit of margin so it's not hugging the screen
    distance += 2 * cube_size
    # move the world away from the camera
    glTranslatef(0.0, 0.0, -distance)
    # rotate the world to the correct orientation
    glRotatef(5, 1, 0, 0)
    glRotatef(90, 0, 1, 0)


# ------------------ Main loop ------------------

def main():
    slices = map2  # game map
    cube_size = 2.0
    spacing = 10.0  # how far apart slices are

    init_pygame_opengl(
        num_slices=len(slices),
        width=800,
        height=600,
        cube_size=cube_size,
        spacing=spacing)

    running = True
    clock = pygame.time.Clock()

    # how far the lane starts toward the camera (in +X)
    lane_x_offset = 10.0
    # movement speed per frame
    speed = 0.15
    # when passed this, jump back to beginning
    reset_distance = (cube_size + spacing)*len(slices) + 15*cube_size

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # move lane toward camera
        lane_x_offset += speed
        if lane_x_offset > reset_distance:
            lane_x_offset = 0.0  # restart from back

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        # move entire lane along -X (toward camera)
        glTranslatef(-lane_x_offset, 0.0, 0.0)
        draw_lane_from_slices(slices, cube_size, spacing)
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

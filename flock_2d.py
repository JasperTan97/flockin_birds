import pygame
import numpy as np
from PIL import Image

from bird_2d import Bird2D

SAVE_ANIMATION = False
ANIMATION_DURATION = 600 # frames

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flocking Simulation with Orientation")

# Create a flock of birds
flock = [Bird2D(np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT)) for _ in range(30)]

if SAVE_ANIMATION:
    frames = []
    frame_count = 0

# Main game loop
clock = pygame.time.Clock()
running = True
while running:
    WINDOW.fill((0, 0, 0))  # Clear the screen

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update and draw each bird
    for bird in flock:
        bird.flock(flock) 
        bird.update()
        bird.edges(WIDTH, HEIGHT)
        bird.draw(WINDOW)

    # Refresh the display
    pygame.display.flip()
    clock.tick(60)  # Limit to 60 FPS

    if SAVE_ANIMATION:
        # Capture the frame
        frame_array = pygame.surfarray.array3d(WINDOW)  # Capture RGB data
        frame_array = np.transpose(frame_array, (1, 0, 2))  # Transpose for PIL format
        frames.append(Image.fromarray(frame_array))
        frame_count += 1
        if frame_count > ANIMATION_DURATION:  
            running = False


pygame.quit()

if SAVE_ANIMATION:
    frames[0].save(
        "flocking_simulation.gif",
        save_all=True,
        append_images=frames[1:],  # Add the rest of the frames
        duration=1000 / 60,  # Duration per frame in milliseconds
        loop=0  # Loop forever
    )
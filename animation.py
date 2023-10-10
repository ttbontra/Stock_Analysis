import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Initialize the grid dimensions and create a 3D grid
grid_dim = 25
grid = np.zeros((grid_dim, grid_dim, grid_dim))

# Initialize the snake's position
snake_pos = [(5, 5, 5)]

# Function to update the grid
def update_grid(grid, snake_pos):
    grid.fill(0)
    for x, y, z in snake_pos:
        if 0 <= x < grid_dim and 0 <= y < grid_dim and 0 <= z < grid_dim:
            grid[x, y, z] = 1

# Function to update the snake's position
def update_snake(snake_pos, direction):
    head_x, head_y, head_z = snake_pos[-1]
    new_head = None
    if direction == 'up':
        new_head = (head_x - 1, head_y, head_z)
    elif direction == 'down':
        new_head = (head_x + 1, head_y, head_z)
    elif direction == 'left':
        new_head = (head_x, head_y - 1, head_z)
    elif direction == 'right':
        new_head = (head_x, head_y + 1, head_z)
    elif direction == 'forward':
        new_head = (head_x, head_y, head_z + 1)
    elif direction == 'backward':
        new_head = (head_x, head_y, head_z - 1)

    # Check if the new head position is within the grid boundaries
    if all(0 <= i < grid_dim for i in new_head):
        snake_pos.append(new_head)
    return snake_pos

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xticks(np.arange(0, grid_dim, 1))
ax.set_yticks(np.arange(0, grid_dim, 1))
ax.set_zticks(np.arange(0, grid_dim, 1))

# Animation function
def animate(frame):
    global snake_pos
    directions = ['up', 'down', 'left', 'right', 'forward', 'backward']
    direction = np.random.choice(directions)
    snake_pos = update_snake(snake_pos, direction)
    update_grid(grid, snake_pos)
    ax.clear()
    x, y, z = zip(*snake_pos)
    ax.scatter(x, y, z, c='blue', s=20)
    ax.plot(x, y, z, c='blue')
    ax.set_xticks(np.arange(0, grid_dim, 1))
    ax.set_yticks(np.arange(0, grid_dim, 1))
    ax.set_zticks(np.arange(0, grid_dim, 1))

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=100)

# Save and display the animation
#ani.save('neural_network_animation_3D_with_trail_and_connections.mp4', writer='ffmpeg')
plt.show()
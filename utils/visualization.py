import matplotlib.pyplot as plt
import numpy as np

def visualize(grid, fire_map, civilians):
    cmap = {
        0: "white",
        1: 'lightblue',
        2: 'green',
        3: 'gray',
        4: 'yellow',
        5: 'red'
    }

    display_grid = np.zeros_like(grid, dtype='<U10')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            display_grid[i, j] = cmap[grid[i, j]]

    for i in range(fire_map.shape[0]):
        for j in range(fire_map.shape[1]):
            if fire_map[i, j]:
                display_grid[i, j] = 'orange'

    for civ in civilians:
        i,j = civ['pos']
        display_grid[i, j] = 'black' if civ['health'] > 0 else 'brown'

    fig, ax = plt.subplots()

    for i in range(display_grid.shape[0]):
        for j in range(display_grid.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=display_grid[i, j]))

    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
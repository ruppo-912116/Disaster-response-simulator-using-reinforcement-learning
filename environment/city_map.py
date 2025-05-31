import numpy as np

# Zone constants
EMPTY = 0
HOME = 1
HOSPITAL = 2
ROAD = 3
SHELTER = 4
FIRE_STATION = 5

def create_city_grid(width, height):
    grid = np.zeros((height, width), dtype=int) # initialize 2D grid with zeroes

    # manually or randomly place zones
    grid[1:3, 1:3] = HOME
    grid[5:6, 5:6] = HOSPITAL
    grid[3:4, 7:9] = ROAD
    grid[8:9, 2:4] = SHELTER
    grid[0:1, 0:1] = FIRE_STATION
    return grid

def spread_fire(grid, fire_map):
    new_fire_map = fire_map.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if fire_map[i, j]:
                # check adjacent grid cells to see home is present
                # if home is present, spread fire to the home
                # fire does not spread diagonally currently
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == HOME:
                        new_fire_map[ni, nj] = 1 # fire spreads
    return new_fire_map



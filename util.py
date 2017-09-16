import numpy as np
import pdb

def in_grid(grid, loc):
    return 0 <= loc[0] < grid.shape[0] and 0 <= loc[1] < grid.shape[1]

def allowable_moves(grid, x, y):
    return diag_moves(grid, x, y) + rook_moves(grid, x, y)

def diag_moves(grid, x, y):
    l_moves = []
    r_moves = []
    t = max(grid.shape)

    for i in range(1, t):
        curr_loc = (x + i , y - i)
        if in_grid(grid, curr_loc):
            l_moves.append(curr_loc)
        else:
            break

    for i in range(1, t):
        curr_loc = (x - i , y + i)
        if in_grid(grid, curr_loc):
            l_moves.append(curr_loc)
        else:
            break

    for i in range(1, t):
        curr_loc = (x + i , y + i)
        if in_grid(grid, curr_loc):
            r_moves.append(curr_loc)
        else:
            break

    for i in range(1, t):
        curr_loc = (x - i , y - i)
        if in_grid(grid, curr_loc):
            r_moves.append(curr_loc)
        else:
            break

    return l_moves + r_moves

def rook_moves(grid, x, y):
    v_moves = [(x, i) for i in range(grid.shape[0]) if i != y]
    h_moves = [(i, y) for i in range(grid.shape[1]) if i != x]
    return h_moves + v_moves

def inbetween(grid, x, y):
    btw = []

    # grab horizontal between vals
    for start in range(x):
        for end in range(x+1, grid.shape[0]):
            btw.append(grid[start, y] + grid[end, y])

    # grab vertical between vals
    for start in range(y):
        for end in range(y+1, grid.shape[1]):
            btw.append(grid[x, start] + grid[x, end])

    return np.array(btw)

def make_between_cache(grid):
    cache = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # iterate over the grid then
            sandwich_vals = inbetween(grid, i, j)
            cache[(i, j)] = sandwich_vals

    return cache

def make_nbrs_dict(grid):
    nbrs = {}
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            nbrs[(x, y)] = set(allowable_moves(grid, x, y))

    return nbrs

def gen_neighbors(node, grid):
    x, y, t = node
    max_x = grid.shape[0]
    max_y = grid.shape[1]
    nbrs = []

    # square neighbors are of the form
    reachable = allowable_moves(grid, x, y)
    btw_cache = make_between_cache(grid)

    for loc in reachable:
        if between_square(grid, loc, t):
            nbrs.append((loc, t + SQ_PENALTY))
        else:
            nbrs.append((loc, t + NON_SQ_PENALTY))
    return nbrs

'''
grid = [
    [ 1,  6,  9, 19, 29, 10, 16,  0],
    [21, 20, 28,  4,  9, 26,  7, 14],
    [27, 31,  8, 11, 19,  4, 12, 21],
    [12, 17, 24, 26,  3, 24, 25,  5],
    [ 0, 31,  1, 11, 22,  7, 16, 20],
    [10, 15, 18, 28,  2, 18, 27,  6],
    [17, 22, 30,  3, 13, 25,  2, 14],
    [8 ,  5, 13, 23, 29, 15, 23, 30],
]
grid = np.array(grid).T
print grid.T[::-1]
print allowable_moves(grid, 0, 0)
print inbetween(grid, 0,0)
print inbetween(grid, 7,0)
print inbetween(grid, 7,7)
print inbetween(grid, 1, 2)
pdb.set_trace()
'''

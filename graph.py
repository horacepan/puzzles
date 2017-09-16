import pdb
import numpy as np
from util import allowable_moves, make_nbrs_dict, make_between_cache
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

NON_SQ_PENALTY = 5
SQ_PENALTY = 1
SQUARES = [i*i for i in range(8)]

class Node:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.edges = []

    def add_edge(self, other, cost=None):
        self.edges.append((other, cost))

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, *node):
        self.nodes[node] = Node(*node)

    def add_directed_edge(self, src, dst, cost):
        self.nodes[src].add_edge(dst, cost)

def between_square(grid, loc, between_cache, t):
    # in between cache we will have an np array of all the between values
    if loc not in between_cache:
        raise ValueError

    between_vals = between_cache.get(loc) - t
    return any(map(lambda x: x in SQUARES, between_vals))

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

def edge_cost(grid, src, dst):
    (_, _, t) = src
    (x, y, _) = dst
    return grid[x, y] - t

def make_graph(grid, max_iter=30):
    max_x = grid.shape[0]
    max_y = grid.shape[1]
    nbrs = make_nbrs_dict(grid)
    btw_cache = make_between_cache(grid)
    g = Graph()

    for x in range(max_x):
        for y in range(max_y):
            for t in range(max_iter):
                if (x, y, t) == (7, 7, 29):
                    print "ENTERED"
                g.add_node(x, y, t)

    print 'num nodes:', len(g.nodes)

    for src in g.nodes:
        # need the x, y, t of the node
        # link it to the x', y', t+1/t+5
        src = (x, y, t)
        neighbors = nbrs[(x, y)]
        for (x1, y1) in neighbors:
            if between_square(grid, (x1, y1), btw_cache, t):
                dst = (x1, y1, t+1)
            else:
                dst = (x1, y1, t+5)

            cost = edge_cost(grid, src, dst)
            g.add_directed_edge(src, dst, cost)

make_graph(grid)

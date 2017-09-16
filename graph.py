import pdb
import numpy as np
import Queue
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
        self.edges = {}

    def add_edge(self, other_node, cost=None):
        self.edges[other_node] = cost

    def __repr__(self):
        return 'Node({}, {}, {})'.format(self.x, self.y, self.t)
        #return self.chess_repr()

    def chess_repr(self):
        x = 'abcdefgh'[self.x]
        y = self.y + 1
        return '{}{}'.format(x, y)

class Graph:
    def __init__(self):
        self.nodes = {}
        self._ordered_nodes = []
        self.num_edges = 0

    def add_node(self, *node_params):
        # assume shit is added in topological order
        node = Node(*node_params)
        self.nodes[node_params] = node
        self._ordered_nodes.append(node)

    def add_directed_edge(self, src, dst, cost):
        dst_node = self.nodes[dst]
        self.nodes[src].add_edge(dst_node, cost)
        self.num_edges += 1
    def edge_cost(self, src, dst):
        src_node = self.nodes[src]
        dst_node = self.nodes[dst]
        return src_node.edges[dst_node]

def between_square(grid, loc, between_cache, t):
    # in between cache we will have an np array of all the between values
    if loc not in between_cache:
        raise ValueError

    between_vals = between_cache.get(loc) - 2*t
    if loc == (1, 4) and t == 8:
        pass
        #print between_vals
        #pdb.set_trace()
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

def edge_cost(grid, loc):
    (x, y, t) = loc
    return grid[x, y] - t

def make_graph(grid, max_iter=31):
    max_x = grid.shape[0]
    max_y = grid.shape[1]
    nbrs = make_nbrs_dict(grid)
    btw_cache = make_between_cache(grid)
    g = Graph()

    for t in range(max_iter):
        for x in range(max_x):
            for y in range(max_y):
                g.add_node(x, y, t)

    for src in g.nodes:
        # need the x, y, t of the node # link it to the x', y', t+1/t+5
        x, y, t = src
        neighbors = nbrs[(x, y)]
        for (x1, y1) in neighbors:
            if (x, y, t) == (0, 0, 0):
                dst = (x1, y1, 0)
            elif between_square(grid, (x, y), btw_cache, t):
                dst = (x1, y1, t+1)
            else:
                dst = (x1, y1, t+5)


            if (x, y, t) == (1, 4, 8) and (x1, y1) == (4, 7):
                pass
                #pdb.set_trace()

            cost = edge_cost(grid, dst)
            if dst[2] > 30:
                continue
            g.add_directed_edge(src, dst, cost)
    return g, btw_cache

def backtrace(grid, pred_dict, dst):
    curr_node = dst
    path = []
    vals = []

    while curr_node is not None:
        x, y, t = curr_node.x, curr_node.y, curr_node.t
        path.append(curr_node)
        if (x, y) == (0,0):
            vals.append(0)
        else:
            vals.append(grid[curr_node.x, curr_node.y] - t)
        curr_node = pred_dict[curr_node]

    return path[::-1], vals[::-1]

def single_source_max_paths(graph, src):
    costs = {}
    node = graph.nodes[src]
    costs[node] = 0
    prio_q = Queue.PriorityQueue()

    curr_layer = [node]
    prio_q.put((0, node))
    num_pops = 0
    visited = set()
    predecessors = { node: None for node in graph._ordered_nodes }

    # TODO: Dont really need a prio queue since we have a topological ordering
    while not prio_q.empty():
        (_, curr_node) = prio_q.get()
        num_pops += 1

        for (nbr, cost) in curr_node.edges.iteritems():
            if (nbr not in costs) or (costs[nbr] < costs[curr_node] + cost):
                costs[nbr] = costs[curr_node] + cost
                predecessors[nbr] = curr_node

            if nbr not in visited:
                prio_q.put((nbr.t, nbr))
                visited.add(nbr)

    return costs, predecessors

g, btw_cache = make_graph(grid)
costs, predecessors = single_source_max_paths(g, (0, 0, 0))

best_cost = 0
best_node = None
for t in range(1, 31):
    node = g.nodes[(7, 7, t)]
    if node not in costs:
        continue

    if costs[node] > best_cost:
        best_cost = costs[node]
        best_node = g.nodes[(7,7,t)]

print 'Max sum path {} via:'.format(best_cost)
path, vals = backtrace(grid, predecessors, best_node)
chess_path = map(lambda x: x.chess_repr(), path)
print chess_path
print vals

import sys
import numpy as np
import pdb
import random
import math
from util import in_grid, allowable_moves
'''
9/15/16
https://www.janestreet.com/puzzles/current-puzzle/
Square Run
A queen is located at a1 and wishes to travel to h8 via a series of one or more moves.
(These must be legal queen's moves.)

After each move, the numbers on each of the squares change.

If the move is between two spaces which sum to a perfect square, every number on the board decreases
by 1 after the move. Otherwise, each number decreases by 5. (The queen may stop on a square more
than once.)

What is the largest sum you can obtain from the squares you visit over each move in your journey?
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
squares = set([0, 1, 4, 9, 16, 25, 36, 49])

def argmax(key_lst, val_lst):
    _max = val_lst[0]
    _key = key_lst[0]
    for key, val in zip(key_lst, val_lst):
        if val > _max:
            _max = val
            _key = key
    return _key

def is_sq(val, tol=1e-5):
    if val < 0:
        return False

    sqrt = round(math.sqrt(val))
    return abs((sqrt*sqrt) - val) < tol

def is_square(val):
    return val in squares

def in_grid(grid, loc):
    return 0 <= loc[0] < grid.shape[0] and 0 <= loc[1] < grid.shape[1]

def btw_sq(grid, loc):
    x, y = loc
    sq = False

    if in_grid(grid, (x-1, y)) and in_grid(grid, (x+1, y)):
        v1 = grid[x-1, y] + grid[x+1, y]
        sq = sq or is_sq(v1)

    if in_grid(grid, (x, y+1)) and in_grid(grid, (x, y-1)):
        v2 = grid[x, y+1] + grid[x, y-1]
        sq = sq or is_sq(v2)

    return sq

class Agent(object):
    def __init__(self, grid, policy):
        self._init_grid = grid.copy()
        self.max_x = grid.shape[0]
        self.max_y = grid.shape[1]
        self.grid = grid
        self.x = 0
        self.y = 0
        self.total = 0
        self.num_moves = 0
        self.path = [(self.x, self.y)]
        self.rewards = []
        self.deltas = []
        self.policy = policy

    def __other_call__(self, *inputs):
        assert len(inputs) == 2
        assert isinstance(inputs[0], str)
        assert isinstance(inputs[1], int)
        assert 'a' <= inputs[0] <= 'h'
        assert 1 <= inputs[1] <= 8

        x = 8 - inputs[1]
        y = ord(inputs[0]) - ord('a')
        print '[{},{}] is: {}'.format(inputs[0], inputs[1], self.grid[x][y])

    def __call__(self, *inputs):
        return grid[inputs[0], inputs[1]]

    def update(self):
        loc = (self.x, self.y)
        delta = -1 if btw_sq(self.grid, loc) else -5
        update_func = np.vectorize(lambda x: x + delta)
        self.grid = update_func(self.grid)

    def step(self):
        possible_moves = self.allowable_moves(self.grid, self.x, self.y)

        if self.policy == 'greedy':
            self.greedy_move(possible_moves)
        elif self.policy == 'greedy_rand':
            self.greedy_rand_move(possible_moves)
        elif self.policy == 'random':
            self.random_move(possible_moves)
        elif self.policy == 'manual':
            self.manual_move(possible_moves)
        else:
            pass

    def greedy_move(self, moves):
        vals = map(lambda x: self.grid[x], moves)
        loc = argmax(moves, vals)
        self.move(*loc)

    def random_move(self, moves):
        loc = random.choice(moves)
        self.move(*loc)

    def greedy_rand_move(self, moves):
        if random.random() > 0.8:
            self.random_move(moves)

        # filter onto the ones that are squares
        loc = (self.x, self.y)
        sqs = filter(lambda m: btw_sq(self.grid, m), moves)
        if len(sqs) == 0:
            self.random_move(moves)
        else:
            self.greedy_move(sqs)

    def manual_move(self, moves):
        move_vals = [self.grid[x] for x in moves]
        sq_moves = filter(lambda m: btw_sq(self.grid, m), moves)
        sq_vals = [self.grid[x] for x in sq_moves]
        self.pp_grid()
        print "Current total: {}".format(self.total)
        print "Allowable moves are: {}".format(sorted(zip(move_vals, moves)))
        print "Allowable square moves are: {}".format(sorted(zip(sq_vals, sq_moves)))
        inputs = raw_input("Enter a move:\n")
        x, y = map(int, inputs.split())
        self.move(x, y)

    def move(self, x, y):
        self.x = x
        self.y = y
        self.total += self.grid[x, y]
        self.num_moves += 1
        self.path.append((x, y))
        self.rewards.append(self.grid[x, y])
        self.update()
        self.deltas.append(self.rewards[-1] - self.grid[x, y])

    def pp_grid(self):
        # print the grid in the way it appears as a chessboard
        print self.grid.T[::-1]

    def run(self):
        while not self.done():
            self.step()

    def done(self):
        return self.num_moves > 30 or (self.grid < 0).all()

    def reset(self):
        self.x = 0
        self.y = 0
        self.grid = self._init_grid
        self.num_moves=  0
        self.total = 0
        self.path = [(self.x, self.y)]
        self.rewards = []
        self.deltas = []

def main():
    greedy_agent = Agent(grid, 'greedy')
    greedy_rand_agent = Agent(grid, 'greedy_rand')
    random_agent = Agent(grid, 'random')

    greedy_agent.run()
    greedy_rand_agent.run()
    random_agent.run()

    best = 0
    runs = int(sys.argv[1])
    vals = []
    for i in range(runs):
        greedy_rand_agent.run()
        best = max(greedy_rand_agent.total, best)
        vals.append(greedy_rand_agent.total)
        greedy_rand_agent.reset()

    print "Best for {} trials: {}".format(runs, best)

def manual_play():
    agent = Agent(grid, 'manual')
    agent.run()
    print agent.total
    print agent.path
    print len(agent.path)

if __name__ == '__main__':
    manual_play()

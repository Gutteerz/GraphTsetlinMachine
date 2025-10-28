# hex_gtm/selfplay.py
import numpy as np
from .hex_rules import EMPTY, X, O, legal_moves, apply_move, opposite, winner, random_playout

def collect_positions(board, to_move):
    seq = []
    b = board.copy()
    p = to_move
    while True:
        seq.append((b.copy(), p))
        w = winner(b)
        if w or not legal_moves(b):
            break
        # epsilon-greedy random for data diversity
        moves = legal_moves(b)
        mv = moves[np.random.randint(len(moves))]
        b = apply_move(b, mv, p)
        p = opposite(p)
    # Backfill final result
    w = winner(b)
    res = 0
    if w == to_move: res = 1
    return [(pos, tm, res) for (pos, tm) in seq]

def make_dataset(n=7, games=2000, seed=1):
    np.random.seed(seed)
    data = []
    for _ in range(games):
        board = np.zeros((n, n), dtype=np.int8)
        # random first player and a few random opening moves
        to_move = X if np.random.rand() < 0.5 else O
        data += collect_positions(board, to_move)
    return data

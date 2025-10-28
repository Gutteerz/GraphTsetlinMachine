# hex_gtm/hex_rules.py
import numpy as np
from collections import deque

EMPTY, X, O = 0, 1, 2

def legal_moves(board):
    return list(zip(*np.where(board == EMPTY)))

def apply_move(board, move, player):
    i, j = move
    nb = board.copy()
    nb[i, j] = player
    return nb

def opposite(p): return X if p == O else O

def winner(board):
    n = board.shape[0]
    # X connects top (row 0) to bottom (row n-1)
    if connected(board, X, [(0, j) for j in range(n)], lambda i,j: i == n-1):
        return X
    # O connects left (col 0) to right (col n-1)
    if connected(board, O, [(i, 0) for i in range(n)], lambda i,j: j == n-1):
        return O
    return 0

def connected(board, player, sources, goal_fn):
    n = board.shape[0]
    seen = set()
    q = deque()
    for s in sources:
        i, j = s
        if board[i, j] == player:
            q.append(s); seen.add(s)
    # 6-neighbors (same as in encode)
    offs = [(-1,0),(0,1),(1,1),(1,0),(0,-1),(-1,-1)]
    while q:
        i, j = q.popleft()
        if goal_fn(i, j): return True
        for di, dj in offs:
            ni, nj = i+di, j+dj
            if 0<=ni<n and 0<=nj<n and board[ni,nj]==player and (ni,nj) not in seen:
                seen.add((ni,nj)); q.append((ni,nj))
    return False

def random_playout(board, to_move):
    b = board.copy()
    p = to_move
    while True:
        w = winner(b)
        if w: return w
        moves = legal_moves(b)
        if not moves: return 0
        mv = moves[np.random.randint(len(moves))]
        b = apply_move(b, mv, p)
        p = opposite(p)

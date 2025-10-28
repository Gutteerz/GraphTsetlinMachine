# play_cli.py
import pickle
import numpy as np
from hex_gtm.hex_rules import X, O, EMPTY, winner, legal_moves, apply_move, opposite

def show(board):
    n = board.shape[0]
    for i in range(n):
        print(" " * i, end="")
        for j in range(n):
            v = board[i,j]
            ch = "·" if v==EMPTY else ("X" if v==X else "O")
            print(ch, end=" ")
        print()

if __name__ == "__main__":
    N = 7
    with open("hex_gtm_model.pkl", "rb") as f:
        model = pickle.load(f)
    board = np.zeros((N,N), dtype=np.int8)
    to_move = X
    while True:
        show(board)
        w = winner(board)
        if w:
            print("Winner:", "X" if w==X else "O")
            break
        if to_move == X:
            mv = model.best_move(board, to_move)
            if mv is None:
                print("No moves.")
                break
            print("AI (X) plays:", mv)
            board = apply_move(board, mv, to_move)
        else:
            print("Your move as O (i j):", end=" ")
            try:
                i, j = map(int, input().split())
            except:
                print("bad input"); continue
            if (i,j) not in legal_moves(board):
                print("illegal"); continue
            board = apply_move(board, (i,j), to_move)
        to_move = opposite(to_move)

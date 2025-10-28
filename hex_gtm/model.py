# hex_gtm/model.py
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from .encode import board_to_graphs
from .hex_rules import legal_moves, apply_move

class HexGTM:
    def __init__(self, n=7, clauses=200, T=50, s=5.0, depth=2, hv_size=1024, hv_bits=4):
        """
        clauses: number of clauses
        T: threshold
        s: specificity
        depth: message-passing rounds (logical depth)
        """
        self.n = n
        self.clf = MultiClassGraphTsetlinMachine(
            number_of_clauses=clauses,
            T=T,
            s=s,
            depth=depth,
        )
        self.hv_size = hv_size
        self.hv_bits = hv_bits

    def _encode_one(self, board, to_move):
        g, gid = board_to_graphs(board, to_move, hv_size=self.hv_size, hv_bits=self.hv_bits)
        return g, gid

    def fit(self, dataset, epochs=30):
        """
        dataset: list of (board, to_move, y) with y in {0,1}
        """
        # Build one Graphs object per sample and train epoch-wise
        X = []
        Y = []
        for board, to_move, y in dataset:
            g, gid = self._encode_one(board, to_move)
            X.append(g); Y.append(np.uint32(y))
        # Simple training loop: GraphTM expects graphs per sample
        self.clf.fit(X, np.array(Y, dtype=np.uint32), epochs=epochs)

    def predict_proba(self, board, to_move):
        g, gid = self._encode_one(board, to_move)
        # returns prob of class 1 (win for side-to-move)
        return float(self.clf.predict_proba([g])[0][1])

    def best_move(self, board, to_move):
        moves = legal_moves(board)
        if not moves:
            return None
        scores = []
        for mv in moves:
            b2 = apply_move(board, mv, to_move)
            prob = self.predict_proba(b2, 3 - to_move)  # after move, opponent to move
            # choose move that minimizes opponent win prob
            scores.append((1.0 - prob, mv))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0][1]

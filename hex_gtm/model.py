# hex_gtm/model.py
import numpy as np
import os, json
try:
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine as GTM
except ImportError:
    from GraphTsetlinMachine.tm import GraphTsetlinMachine as GTM

from .encode import make_empty_batch, init_topology, set_properties
from .hex_rules import legal_moves, apply_move

class HexGTM:
    def __init__(self, n=7, clauses=200, T=50, s=5.0, depth=2, hv_size=1024, hv_bits=4):
        self.n = n
        self.hv_size = hv_size
        self.hv_bits = hv_bits
        try:
            self.clf = GTM(number_of_clauses=clauses, T=T, s=s, message_passing_rounds=depth)
        except TypeError:
            self.clf = GTM(number_of_clauses=clauses, T=T, s=s)

    def fit(self, dataset, epochs=30):
        m = len(dataset)
        graphs = make_empty_batch(m, hv_size=self.hv_size, hv_bits=self.hv_bits)
        init_topology(graphs, self.n)  # build nodes+edges once for all m graphs
        Y = np.zeros(m, dtype=np.uint32)
        for i, (board, to_move, y) in enumerate(dataset):
            set_properties(graphs, i, board, to_move)
            Y[i] = np.uint32(y)
        self.clf.fit(graphs, Y, epochs=epochs)

    def predict_proba(self, board, to_move):
        graphs = make_empty_batch(1, hv_size=self.hv_size, hv_bits=self.hv_bits)
        init_topology(graphs, self.n)
        set_properties(graphs, 0, board, to_move)
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(graphs)[0]  # [p0, p1]
            return float(p[1])
        else:
            pred = int(self.clf.predict(graphs)[0])
            return 1.0 if pred == 1 else 0.0

    def best_move(self, board, to_move):
        moves = legal_moves(board)
        if not moves:
            return None
        scores = []
        for mv in moves:
            b2 = apply_move(board, mv, to_move)
            opp_win = self.predict_proba(b2, 3 - to_move)
            scores.append((1.0 - opp_win, mv))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0][1]

        def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # 1) save hyperparams
        meta = {
            "n": self.n,
            "hv_size": self.hv_size,
            "hv_bits": self.hv_bits,
            # We can't reliably introspect clauses/T/s back from 0.3.3, so store what we passed in:
            # If you want, keep them on self.* at __init__ time.
        }
        # if you kept these on self in __init__, include them:
        for k in ("clauses","T","s"):
            if hasattr(self, k): meta[k] = getattr(self, k)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        # 2) save classifier with native method if available
        clf_bin = os.path.join(path, "clf.bin")
        if hasattr(self.clf, "save"):
            self.clf.save(clf_bin)
        else:
            # No safe way to serialize. Leave a marker.
            with open(os.path.join(path, "README.txt"), "w") as f:
                f.write("This GTM build has no .save(); re-train to recreate the classifier.\n")

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        # pass stored hparams; fall back if some missing
        n       = meta.get("n", 7)
        hv_size = meta.get("hv_size", 1024)
        hv_bits = meta.get("hv_bits", 4)
        clauses = meta.get("clauses", 200)
        T       = meta.get("T", 50)
        s       = meta.get("s", 5.0)

        model = cls(n=n, clauses=clauses, T=T, s=s, hv_size=hv_size, hv_bits=hv_bits)
        clf_bin = os.path.join(path, "clf.bin")
        if hasattr(model.clf, "load") and os.path.exists(clf_bin):
            model.clf.load(clf_bin)
        else:
            raise RuntimeError("No saved classifier found. Re-train the model.")
        return model

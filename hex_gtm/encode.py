# hex_gtm/encode.py
import numpy as np

# 0.3.3 import (lowercase); fallback for other builds
try:
    from GraphTsetlinMachine.graphs import Graphs
except ImportError:
    from GraphTsetlinMachine.Graphs import Graphs
import numpy as np


EDGE_TYPES = ["NE", "E", "SE", "SW", "W", "NW"]
EDGE_SYMS  = EDGE_TYPES + ["E"]  # we use "E" for goal edges
SYMS = [
    "X", "O", "Empty",
    "SideTop", "SideBottom", "SideLeft", "SideRight",
    "ToMoveX", "ToMoveO",
]

def _stamp_signature(graphs, n, bsz):
    # Any stable numpy array works; trainer only checks equality.
    sig = np.array([len(SYMS), len(EDGE_SYMS), n, bsz], dtype=np.int64)
    setattr(graphs, "signature", sig)


def neighbors_axial(i, j, n):
    # NE, E, SE, SW, W, NW
    offs = [(-1,0),(0,1),(1,1),(1,0),(0,-1),(-1,-1)]
    for idx, (di, dj) in enumerate(offs):
        ni, nj = i+di, j+dj
        if 0 <= ni < n and 0 <= nj < n:
            yield idx, ni, nj

def _idx(i, j, n): return i*n + j

def make_empty_batch(batch_size, hv_size=1024, hv_bits=4):
    g = Graphs(
        batch_size,
        symbols=SYMS + EDGE_SYMS,
        hypervector_size=hv_size,
        hypervector_bits=hv_bits,
    )
    setattr(g, "_bsz", batch_size)  # for old versions
    return g

def init_topology(graphs: Graphs, n: int):
    bsz = getattr(graphs, "batch_size", getattr(graphs, "_bsz", 1))

    node_count = n*n + 4
    for gi in range(bsz):
        graphs.set_number_of_graph_nodes(gi, node_count)

    graphs.prepare_node_configuration()

    for gi in range(bsz):
        for k in range(n*n):
            graphs.add_graph_node(gi, f"C{k}", 6)
        graphs.add_graph_node(gi, "XTop",    n*n)
        graphs.add_graph_node(gi, "XBottom", n*n)
        graphs.add_graph_node(gi, "OLeft",   n*n)
        graphs.add_graph_node(gi, "ORight",  n*n)

    graphs.prepare_edge_configuration()

    for gi in range(bsz):
        for i in range(n):
            for j in range(n):
                a = _idx(i, j, n)
                for etype_idx, ni, nj in neighbors_axial(i, j, n):
                    b = _idx(ni, nj, n)
                    graphs.add_graph_node_edge(gi, f"C{a}", f"C{b}", EDGE_TYPES[etype_idx])
        for j in range(n):
            graphs.add_graph_node_edge(gi, "XTop",                f"C{_idx(0, j, n)}",    "E")
            graphs.add_graph_node_edge(gi, f"C{_idx(n-1, j, n)}", "XBottom",              "E")
        for i in range(n):
            graphs.add_graph_node_edge(gi, "OLeft",               f"C{_idx(i, 0, n)}",    "E")
            graphs.add_graph_node_edge(gi, f"C{_idx(i, n-1, n)}", "ORight",               "E")

    _stamp_signature(graphs, n, bsz)



def set_properties(graphs: Graphs, gi: int, board: np.ndarray, to_move: int):
    """
    Assign node properties for ONE sample graph `gi`.
    Call AFTER init_topology().
    """
    n = board.shape[0]
    for i in range(n):
        for j in range(n):
            k = _idx(i, j, n)
            v = int(board[i, j])
            graphs.add_graph_node_property(gi, f"C{k}",
                "X" if v==1 else ("O" if v==2 else "Empty"))
            if i == 0:      graphs.add_graph_node_property(gi, f"C{k}", "SideTop")
            if i == n - 1:  graphs.add_graph_node_property(gi, f"C{k}", "SideBottom")
            if j == 0:      graphs.add_graph_node_property(gi, f"C{k}", "SideLeft")
            if j == n - 1:  graphs.add_graph_node_property(gi, f"C{k}", "SideRight")
    mark = "ToMoveX" if to_move == 1 else "ToMoveO"
    for k in range(n*n):
        graphs.add_graph_node_property(gi, f"C{k}", mark)

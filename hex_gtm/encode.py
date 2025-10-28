# hex_gtm/encode.py
import numpy as np
from GraphTsetlinMachine import Graphs

# Edge type names for the 6 hex directions (pointy-top axial-ish)
EDGE_TYPES = ["NE", "E", "SE", "SW", "W", "NW"]

# Node symbols (properties)
SYMS = [
    "X", "O", "Empty",
    "SideTop", "SideBottom", "SideLeft", "SideRight",
    "ToMoveX", "ToMoveO",
]

def neighbors_axial(i, j, n):
    # 6-neighborhood on a rhombus layout
    # offsets: NE, E, SE, SW, W, NW
    offs = [(-1, 0), (0, 1), (1, 1), (1, 0), (0, -1), (-1, -1)]
    for idx, (di, dj) in enumerate(offs):
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            yield idx, ni, nj

def idx(i, j, n):
    return i * n + j

def board_to_graphs(board, to_move, symbols=SYMS, hv_size=1024, hv_bits=4):
    """
    board: np.array shape (n,n) with values: 0=Empty, 1=X, 2=O
    to_move: 1 for X, 2 for O
    Returns: (graphs, graph_id)
    """
    n = board.shape[0]
    node_count = n * n + 4  # +4 virtual goal nodes: X_top, X_bottom, O_left, O_right
    g = Graphs(
        1,
        symbols=symbols + EDGE_TYPES,   # edge types also become symbols
        hypervector_size=hv_size,
        hypervector_bits=hv_bits,
    )
    g.set_number_of_graph_nodes(0, node_count)
    g.prepare_node_configuration()

    # Add real cell nodes with 6 outgoing edges (max)
    # Then 4 goal nodes with many outgoing edges (we’ll over-allocate)
    # Use a simple upper bound: cells: <=6, goals: <=n*n
    for k in range(n * n):
        g.add_graph_node(0, f"C{k}", 6)
    # Goal nodes
    g.add_graph_node(0, "XTop", n * n)
    g.add_graph_node(0, "XBottom", n * n)
    g.add_graph_node(0, "OLeft", n * n)
    g.add_graph_node(0, "ORight", n * n)

    g.prepare_edge_configuration()

    # Add cell-to-cell edges with types
    for i in range(n):
        for j in range(n):
            a = idx(i, j, n)
            for etype_idx, ni, nj in neighbors_axial(i, j, n):
                b = idx(ni, nj, n)
                g.add_graph_node_edge(0, f"C{a}", f"C{b}", EDGE_TYPES[etype_idx])

    # Add goal edges: X connects top↔bottom borders; O connects left↔right borders
    # X goals
    for j in range(n):
        g.add_graph_node_edge(0, "XTop",    f"C{idx(0, j, n)}",    "E")  # type arbitrary
        g.add_graph_node_edge(0, f"C{idx(n-1, j, n)}", "XBottom", "E")
    # O goals
    for i in range(n):
        g.add_graph_node_edge(0, "OLeft",   f"C{idx(i, 0, n)}",    "E")
        g.add_graph_node_edge(0, f"C{idx(i, n-1, n)}", "ORight",   "E")

    # Properties
    # Cell content: X/O/Empty + border tags
    for i in range(n):
        for j in range(n):
            k = idx(i, j, n)
            v = board[i, j]
            if v == 1:
                g.add_graph_node_property(0, f"C{k}", "X")
            elif v == 2:
                g.add_graph_node_property(0, f"C{k}", "O")
            else:
                g.add_graph_node_property(0, f"C{k}", "Empty")
            if i == 0:      g.add_graph_node_property(0, f"C{k}", "SideTop")
            if i == n - 1:  g.add_graph_node_property(0, f"C{k}", "SideBottom")
            if j == 0:      g.add_graph_node_property(0, f"C{k}", "SideLeft")
            if j == n - 1:  g.add_graph_node_property(0, f"C{k}", "SideRight")

    # To-move marker (global via all nodes for simplicity)
    if to_move == 1:
        mark = "ToMoveX"
    else:
        mark = "ToMoveO"
    for k in range(n * n):
        g.add_graph_node_property(0, f"C{k}", mark)

    return g, 0

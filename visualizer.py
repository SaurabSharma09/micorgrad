"""
visualizer.py - Computation graph visualizer for micrograd-explained
=====================================================================
Draws the computation graph of any Value using Graphviz.

Each node shows:  label | data | grad
Each edge shows:  which operation produced this node

Usage:
    from engine import Value
    from visualizer import draw_graph, draw_graph_html

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = (a * b).relu()
    c.backward()

    # Save as PNG or SVG:
    draw_graph(c, filename='my_graph', format='png')

    # Get inline HTML string (for notebooks):
    html = draw_graph_html(c)

Why visualize the computation graph?
    The graph makes backprop concrete. You can see exactly which nodes
    connect, watch data flow forward, and verify gradients flow backward.
    This is what PyTorch's autograd is doing internally — it just hides it.
"""

from graphviz import Digraph


def _build_graph(root):
    """
    Walk the computation graph starting from root and collect
    all nodes and edges.

    Returns:
        nodes : set of all Value objects in the graph
        edges : set of (child, parent) tuples
    """
    nodes = set()
    edges = set()

    def visit(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                visit(child)

    visit(root)
    return nodes, edges


def draw_graph(root, filename='computation_graph', format='svg',
               rankdir='LR', show_grad=True):
    """
    Draw the computation graph and save it to a file.

    Each Value becomes a rectangular node showing:
        label (if set) | data value | gradient

    Each operation (*, +, tanh, etc.) becomes a small oval node.
    Edges connect children → operation → output.

    Args:
        root       : the output Value to trace backward from
        filename   : output file name (no extension)
        format     : 'svg', 'png', or 'pdf'
        rankdir    : 'LR' = left-to-right, 'TB' = top-to-bottom
        show_grad  : if False, hides gradient from node labels

    Returns:
        dot : the Digraph object (can call dot.view() to open it)
    """
    nodes, edges = _build_graph(root)

    dot = Digraph(
        format=format,
        graph_attr={
            'rankdir': rankdir,
            'bgcolor': '#1e1e2e',  # dark background
            'fontname': 'Helvetica',
        }
    )

    for n in nodes:
        # build the label string for this node
        label_parts = []

        # 1. user-provided label (e.g. 'a', 'b', 'loss')
        if n.label:
            label_parts.append(f" {n.label} ")

        # 2. data value — always shown
        label_parts.append(f" data={n.data:.4f} ")

        # 3. gradient — shown if backward has been called
        if show_grad:
            label_parts.append(f" grad={n.grad:.4f} ")

        node_label = " | ".join(label_parts)

        # unique id for this node
        uid = str(id(n))

        # rectangular node for each Value
        dot.node(
            name=uid,
            label="{ " + node_label + " }",
            shape='record',
            style='filled',
            fillcolor='#313244',
            fontcolor='#cdd6f4',
            color='#89b4fa',
            fontname='Courier',
            fontsize='11',
        )

        # if this node was produced by an operation, add an operation node
        if n._op:
            op_uid = uid + n._op  # make it unique per node+op combo

            dot.node(
                name=op_uid,
                label=n._op,
                shape='ellipse',
                style='filled',
                fillcolor='#45475a',
                fontcolor='#f38ba8',
                color='#f38ba8',
                fontname='Helvetica-Bold',
                fontsize='12',
            )

            # operation node → output value node
            dot.edge(
                op_uid, uid,
                color='#a6e3a1',
                penwidth='1.5',
            )

    for child, parent in edges:
        # child value → parent's operation node
        dot.edge(
            str(id(child)),
            str(id(parent)) + parent._op,
            color='#89dceb',
            penwidth='1.2',
        )

    dot.render(filename, cleanup=True)
    print(f"Graph saved to {filename}.{format}")
    return dot


def draw_graph_html(root, rankdir='LR', show_grad=True):
    """
    Return the computation graph as an inline SVG string.
    Perfect for Jupyter notebooks — embed directly in cell output.

    Usage in notebook:
        from IPython.display import HTML
        html = draw_graph_html(loss)
        display(HTML(html))

    Args:
        root      : the output Value to trace from
        rankdir   : 'LR' (left-right) or 'TB' (top-bottom)
        show_grad : include gradient values in node labels

    Returns:
        str : raw SVG string
    """
    nodes, edges = _build_graph(root)

    dot = Digraph(
        format='svg',
        graph_attr={
            'rankdir': rankdir,
            'bgcolor': 'transparent',
            'fontname': 'Helvetica',
        }
    )

    for n in nodes:
        label_parts = []
        if n.label:
            label_parts.append(f" {n.label} ")
        label_parts.append(f" {n.data:.4f} ")
        if show_grad:
            label_parts.append(f" ∇={n.grad:.4f} ")
        node_label = " | ".join(label_parts)

        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ " + node_label + " }",
            shape='record',
            style='filled',
            fillcolor='#dde3f0',
            fontname='Courier',
            fontsize='10',
        )

        if n._op:
            op_uid = uid + n._op
            dot.node(
                name=op_uid,
                label=n._op,
                shape='ellipse',
                style='filled',
                fillcolor='#ffd7b5',
                fontname='Helvetica-Bold',
                fontsize='11',
            )
            dot.edge(op_uid, uid)

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot.pipe(format='svg').decode('utf-8')


def print_graph_text(root, indent=0, visited=None):
    """
    Fallback: print the computation graph as indented text.
    Works without graphviz installed.

    Usage:
        print_graph_text(loss)

    Output example:
        [loss] tanh | data=0.7071 | grad=1.0000
          [+] | data=0.5000 | grad=...
            [a] | data=2.0 | grad=...
            [b] | data=-1.5 | grad=...
    """
    if visited is None:
        visited = set()
    if id(root) in visited:
        return
    visited.add(id(root))

    prefix = "  " * indent
    label = f"[{root.label}] " if root.label else ""
    op = f"{root._op} | " if root._op else ""
    print(f"{prefix}{label}{op}data={root.data:.4f} | grad={root.grad:.4f}")

    for child in root._prev:
        print_graph_text(child, indent + 1, visited)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo — run this file directly to see it work
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from engine import Value

    print("Building example computation graph...")
    print("  a = Value(2.0)")
    print("  b = Value(-3.0)")
    print("  c = Value(10.0)")
    print("  d = a*b + c")
    print("  e = d.tanh()")
    print()

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b + c
    d.label = 'd'
    e = d.tanh()
    e.label = 'e'
    e.backward()

    print("Text representation of the graph:")
    print("─" * 50)
    print_graph_text(e)
    print()

    # Try graphviz output
    try:
        draw_graph(e, filename='/tmp/example_graph', format='svg')
        print("SVG graph saved to /tmp/example_graph.svg")
        print("Open it in a browser to see the visual graph.")
    except Exception as ex:
        print(f"Graphviz not available: {ex}")
        print("Install with: pip install graphviz && apt install graphviz")

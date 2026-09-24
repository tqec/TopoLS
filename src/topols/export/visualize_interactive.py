import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Interactive (Plotly) 3D visualization -- a drag/rotate/zoom-able HTML
# counterpart to visualize.py's static Matplotlib renderer, added for the
# Phase 2 debug-and-optimize pass (see docs/REFACTOR_LOG.md's dated entry).
# Mirrors visualize.py's geometry and color-coding exactly (same cube face
# order, same tqec/S/T/input-output color rules, same split-edge + yellow
# "color transition" collar logic) so the two renderers show identical
# diagrams -- this one just lets you actually grab and rotate it.
# ---------------------------------------------------------------------------

AXIS_COLOR = {
    "X": "red",
    "Z": "blue",
}


def tqec_axis_colors(tqec):
    return {
        "x": AXIS_COLOR[tqec[0]],
        "y": AXIS_COLOR[tqec[1]],
        "z": AXIS_COLOR[tqec[2]],
    }


def needs_color_transition(tqec1, tqec2, edge_axis):
    c1 = tqec_axis_colors(tqec1)
    c2 = tqec_axis_colors(tqec2)
    for axis in {"x", "y", "z"} - {edge_axis}:
        if c1[axis] != c2[axis]:
            return True
    return False


def edge_axis(p1, p2):
    dx, dy, dz = (p2[i] - p1[i] for i in range(3))
    if abs(dx) == 1:
        return "x"
    if abs(dy) == 1:
        return "y"
    if abs(dz) == 1:
        return "z"
    raise ValueError("Invalid edge (not unit length)")


def midpoint(a, b):
    return tuple((a[i] + b[i]) / 2 for i in range(3))


def edge_endpoints(p1, p2, cube_half):
    dx, dy, dz = (p2[i] - p1[i] for i in range(3))
    if abs(dx) == 1:
        return (
            (p1[0] + cube_half * dx, p1[1], p1[2]),
            (p2[0] - cube_half * dx, p2[1], p2[2]),
        )
    if abs(dy) == 1:
        return (
            (p1[0], p1[1] + cube_half * dy, p1[2]),
            (p2[0], p2[1] - cube_half * dy, p2[2]),
        )
    if abs(dz) == 1:
        return (
            (p1[0], p1[1], p1[2] + cube_half * dz),
            (p2[0], p2[1], p2[2] - cube_half * dz),
        )


def get_node_face_colors(node):
    other = node.get("other")
    tqec = node.get("tqec")

    if isinstance(other, dict):
        if other.get("type") in {"input", "output"}:
            return ["gray"] * 6
    if other == "S":
        return ["green"] * 6
    if other == "T":
        return ["purple"] * 6
    if tqec is not None:
        axis_colors = {"X": "red", "Z": "blue"}
        return [
            axis_colors[tqec[0]], axis_colors[tqec[0]],  # +/-X
            axis_colors[tqec[1]], axis_colors[tqec[1]],  # +/-Y
            axis_colors[tqec[2]], axis_colors[tqec[2]],  # +/-Z
        ]
    return ["black"] * 6


class _MeshAccumulator:
    """Collects boxes ("prisms") into one flat vertex/triangle/facecolor set
    of arrays -- a single Plotly Mesh3d trace with N boxes renders far faster
    than N separate traces."""

    def __init__(self):
        self.x, self.y, self.z = [], [], []
        self.i, self.j, self.k = [], [], []
        self.facecolor = []
        # Wireframe edges (the box outline), matching visualize.py's
        # Poly3DCollection(edgecolors="black") on every face. Stored as a
        # single Scatter3d "lines" trace with None separators between
        # disconnected segments, since Mesh3d has no built-in edge outline.
        self.edge_x, self.edge_y, self.edge_z = [], [], []

    def add_prism(self, center, dims, face_colors):
        """dims = (dx, dy, dz) half-extents. face_colors: 6 colors, in the
        same +X,-X,+Y,-Y,+Z,-Z order as visualize.py's draw_prism."""
        cx, cy, cz = center
        dx, dy, dz = dims
        x0, x1 = cx - dx, cx + dx
        y0, y1 = cy - dy, cy + dy
        z0, z1 = cz - dz, cz + dz

        base = len(self.x)
        # 8 corners: 0-3 is the z0 face (order matches -Z quad below), 4-7 is z1
        corners = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        for cxp, cyp, czp in corners:
            self.x.append(cxp)
            self.y.append(cyp)
            self.z.append(czp)

        # Two triangles per face, in visualize.py's +X,-X,+Y,-Y,+Z,-Z order.
        quads = [
            (1, 2, 6, 5),  # +X
            (0, 3, 7, 4),  # -X
            (3, 2, 6, 7),  # +Y
            (0, 1, 5, 4),  # -Y
            (4, 5, 6, 7),  # +Z
            (0, 1, 2, 3),  # -Z
        ]
        for (a, b, c, d), color in zip(quads, face_colors):
            self.i.append(base + a); self.j.append(base + b); self.k.append(base + c)
            self.facecolor.append(color)
            self.i.append(base + a); self.j.append(base + c); self.k.append(base + d)
            self.facecolor.append(color)

        # 12 box edges (bottom face, top face, 4 verticals connecting them).
        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in box_edges:
            xa, ya, za = corners[a]
            xb, yb, zb = corners[b]
            self.edge_x.extend([xa, xb, None])
            self.edge_y.extend([ya, yb, None])
            self.edge_z.extend([za, zb, None])

    def to_mesh3d(self, name, opacity=1.0):
        if not self.x:
            return None
        return go.Mesh3d(
            x=self.x, y=self.y, z=self.z,
            i=self.i, j=self.j, k=self.k,
            facecolor=self.facecolor,
            opacity=opacity,
            name=name,
            flatshading=True,
            showlegend=False,
        )

    def to_wireframe(self, name, linewidth=1.5, color="black"):
        if not self.edge_x:
            return None
        return go.Scatter3d(
            x=self.edge_x, y=self.edge_y, z=self.edge_z,
            mode="lines",
            line=dict(color=color, width=linewidth),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )


def _add_node(acc, pos, size, colors):
    half = size / 2
    acc.add_prism(pos, (half, half, half), colors)


def _add_edge_segment(acc, center, axis, length, thickness, colors):
    l = length / 2
    t = thickness / 2
    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:
        dims = (t, t, l)
    acc.add_prism(center, dims, colors)


def _add_transition_band(acc, center, axis, edge_thickness, band_len, color="yellow", epsilon=0.02):
    t = (edge_thickness + epsilon) / 2
    l = band_len / 2
    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:
        dims = (t, t, l)
    acc.add_prism(center, dims, [color] * 6)


def _add_connected_edge(acc, p1, p2, node1, node2, cube_size):
    cube_half = cube_size / 2
    edge_thickness = cube_size

    s, e = edge_endpoints(p1, p2, cube_half)
    axis = edge_axis(p1, p2)
    mid = midpoint(s, e)
    edge_length = 1 - 2 * cube_half

    has1 = node1["tqec"] is not None
    has2 = node2["tqec"] is not None

    if has1 and has2:
        tqec1 = node1["tqec"]
        tqec2 = node2["tqec"]
        c1 = tqec_axis_colors(tqec1)
        c2 = tqec_axis_colors(tqec2)

        if needs_color_transition(tqec1, tqec2, axis):
            _add_edge_segment(acc, midpoint(s, mid), axis, edge_length / 2, edge_thickness,
                               [c1[a] for a in ["x", "x", "y", "y", "z", "z"]])
            _add_edge_segment(acc, midpoint(mid, e), axis, edge_length / 2, edge_thickness,
                               [c2[a] for a in ["x", "x", "y", "y", "z", "z"]])
            band_len = edge_thickness * 0.25
            _add_transition_band(acc, mid, axis, edge_thickness=edge_thickness,
                                  band_len=band_len, color="yellow", epsilon=edge_thickness * 0.05)
        else:
            _add_edge_segment(acc, mid, axis, edge_length, edge_thickness,
                               [c1[a] for a in ["x", "x", "y", "y", "z", "z"]])

    elif has1 or has2:
        src = node1 if has1 else node2
        c = tqec_axis_colors(src["tqec"])
        _add_edge_segment(acc, mid, axis, edge_length, edge_thickness,
                           [c[a] for a in ["x", "x", "y", "y", "z", "z"]])


def visualize_interactive(nodes, edges, benchmark, cube_size=0.4, pipe_thickness=0.18, out_path=None):
    """Builds the same pipe diagram as visualize.py, as a single draggable/
    rotatable/zoomable HTML file (Plotly, no server needed -- open directly
    in a browser). Returns the output path."""
    node_acc = _MeshAccumulator()
    label_xs, label_ys, label_zs, label_text = [], [], [], []
    for node_id, node in nodes.items():
        face_colors = get_node_face_colors(node)
        _add_node(node_acc, node["position"], cube_size, face_colors)
        # Node IDs, offset above each cube so the text does not sink into the
        # coloured mesh. Synthetic `path_*` stubs are skipped -- they are not
        # real embedded nodes and labelling them makes the view unreadable.
        if isinstance(node_id, str) and node_id.startswith("path_"):
            continue
        x, y, z = node["position"]
        label_xs.append(x); label_ys.append(y); label_zs.append(z + cube_size * 1.5)
        label_text.append(f"{node_id}")

    edge_acc = _MeshAccumulator()
    for (n1, n2) in edges:
        _add_connected_edge(edge_acc, nodes[n1]["position"], nodes[n2]["position"], nodes[n1], nodes[n2], cube_size)

    traces = []
    node_mesh = node_acc.to_mesh3d("nodes")
    if node_mesh is not None:
        traces.append(node_mesh)
    edge_mesh = edge_acc.to_mesh3d("edges")
    if edge_mesh is not None:
        traces.append(edge_mesh)
    node_wire = node_acc.to_wireframe("node edges")
    if node_wire is not None:
        traces.append(node_wire)
    edge_wire = edge_acc.to_wireframe("pipe edges")
    if edge_wire is not None:
        traces.append(edge_wire)

    traces.append(go.Scatter3d(
        x=label_xs, y=label_ys, z=label_zs,
        mode="text",
        text=label_text,
        textfont=dict(size=11, color="black"),
        textposition="top center",
        hoverinfo="text",
        name="node labels",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{benchmark} pipe diagram",
        scene=dict(
            aspectmode="data",
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    if out_path is None:
        out_path = f"result/visualization/{benchmark}_interactive.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path

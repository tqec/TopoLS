"""Static Matplotlib rendering of a compiled pipe diagram (cubes coloured by
boundary type, S/T gates, ports, and yellow colour-transition collars).
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

# Face colour per boundary type.
AXIS_COLOR = {
    "X": "red",
    "Z": "blue"
}

def tqec_axis_colors(tqec):
    """Face colour along each axis for a TQEC cube type such as `"XZZ"`."""
    return {
        "x": AXIS_COLOR[tqec[0]],
        "y": AXIS_COLOR[tqec[1]],
        "z": AXIS_COLOR[tqec[2]],
    }

def needs_color_transition(tqec1, tqec2, edge_axis):
    """True iff two cubes joined by a pipe along `edge_axis` differ in colour
    on a face perpendicular to the pipe, i.e. the pipe carries a Hadamard
    (drawn as a yellow collar)."""
    c1 = tqec_axis_colors(tqec1)
    c2 = tqec_axis_colors(tqec2)

    for axis in {"x", "y", "z"} - {edge_axis}:
        if c1[axis] != c2[axis]:
            return True

    return False

def draw_transition_band(ax, center, axis, edge_thickness, band_len, color="yellow", epsilon=0.02):
    """
    Draw a thin yellow 'collar band' wrapped around the pipe at the edge midpoint.

    axis: 'x'/'y'/'z' direction of the edge
    edge_thickness: pipe thickness (should match cube_size if you want same thickness)
    band_len: thickness of the band along the edge axis (small)
    epsilon: small expansion so the band is visible on top of the pipe
    """
    t = (edge_thickness + epsilon) / 2
    l = band_len / 2

    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:  # "z"
        dims = (t, t, l)

    draw_prism(ax, center, dims, [color] * 6, edgecolor=color, lw=0.0)

def edge_axis(p1, p2):
    """Axis (`"x"`, `"y"` or `"z"`) of the unit step from `p1` to `p2`."""
    dx, dy, dz = (p2[i] - p1[i] for i in range(3))
    if abs(dx) == 1:
        return "x"
    if abs(dy) == 1:
        return "y"
    if abs(dz) == 1:
        return "z"
    raise ValueError("Invalid edge (not unit length)")

def midpoint(a, b):
    """Midpoint of two 3D points."""
    return tuple((a[i] + b[i]) / 2 for i in range(3))

def draw_prism(ax, center, dims, face_colors, edgecolor="black", lw=0.2):
    """Draw a box of half-extents `dims` with six face colours in the order
    +X, -X, +Y, -Y, +Z, -Z."""
    x, y, z = center
    dx, dy, dz = dims

    x0, x1 = x - dx, x + dx
    y0, y1 = y - dy, y + dy
    z0, z1 = z - dz, z + dz

    faces = [
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # +X
        [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],  # -X
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # +Y
        [(x0,y0,z0),(x0,y0,z1),(x1,y0,z1),(x1,y0,z0)],  # -Y
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # +Z
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # -Z
    ]

    for f, c in zip(faces, face_colors):
        ax.add_collection3d(
            Poly3DCollection([f], facecolors=c, edgecolors=edgecolor, linewidths=lw)
        )

def draw_node(ax, pos, size, colors):
    """Draw a cube of side `size`."""
    half = size / 2
    draw_prism(ax, pos, (half, half, half), colors)

def draw_edge(ax, center, axis, length, thickness, colors):
    """Draw one pipe segment along `axis`."""
    l = length / 2
    t = thickness / 2

    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:
        dims = (t, t, l)

    draw_prism(ax, center, dims, colors)

def draw_connected_edge(
    ax,
    p1, p2,
    node1, node2,
    cube_size
):
    """Draw the pipe between two adjacent nodes.

    Both typed: one pipe in their common colours, or two half pipes with a
    yellow collar where the colours differ. One typed (S/T or port stub):
    the pipe takes that node's colours. Neither typed: nothing is drawn.
    """
    cube_half = cube_size / 2
    edge_thickness = cube_size

    # 1. compute edge geometry (you are correct now)
    s, e = edge_endpoints(p1, p2, cube_half)
    axis = edge_axis(p1, p2)
    mid = midpoint(s, e)
    edge_length = 1 - 2 * cube_half

    # 2. which side has tqec
    has1 = node1["tqec"] is not None
    has2 = node2["tqec"] is not None

    # case A, both sides have tqec
    if has1 and has2:
        tqec1 = node1["tqec"]
        tqec2 = node2["tqec"]

        c1 = tqec_axis_colors(tqec1)
        c2 = tqec_axis_colors(tqec2)

        if needs_color_transition(tqec1, tqec2, axis):
            # left half aligned to node1
            draw_edge(ax, midpoint(s, mid), axis, edge_length/2, edge_thickness,
                    [c1[a] for a in ["x","x","y","y","z","z"]])

            # right half aligned to node2
            draw_edge(ax, midpoint(mid, e), axis, edge_length/2, edge_thickness,
                    [c2[a] for a in ["x","x","y","y","z","z"]])

            # yellow collar band at the center
            band_len = edge_thickness * 0.25
            draw_transition_band(
                ax, mid, axis,
                edge_thickness=edge_thickness,
                band_len=band_len,
                color="yellow",
                epsilon=edge_thickness * 0.05
            )

        else:
            # same color, draw single edge
            draw_edge(
                ax,
                mid,
                axis,
                edge_length,
                edge_thickness,
                [c1[a] for a in ["x","x","y","y","z","z"]]
            )

    # case B, only one side has tqec
    elif has1 or has2:
        src = node1 if has1 else node2
        c = tqec_axis_colors(src["tqec"])

        draw_edge(
            ax,
            mid,
            axis,
            edge_length,
            edge_thickness,
            [c[a] for a in ["x","x","y","y","z","z"]]
        )

    else:
        return

def edge_endpoints(p1, p2, cube_half):
    """Start and end of the visible pipe between two cubes (cube faces excluded)."""
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
    """Six face colours: grey for input/output ports, green for S, purple for
    T, otherwise red/blue by the cube's TQEC type."""
    other = node.get("other")
    tqec = node.get("tqec")

    # Priority 1: input / output
    if isinstance(other, dict):
        if other.get("type") in {"input", "output"}:
            return ["gray"] * 6

    # Priority 2: S / T
    if other == "S":
        return ["green"] * 6
    if other == "T":
        return ["purple"] * 6

    # Priority 3: tqec-based coloring
    if tqec is not None:
        axis_colors = {
            "X": "red",
            "Z": "blue"
        }
        return [
            axis_colors[tqec[0]], axis_colors[tqec[0]],  # ±X
            axis_colors[tqec[1]], axis_colors[tqec[1]],  # ±Y
            axis_colors[tqec[2]], axis_colors[tqec[2]],  # ±Z
        ]

    # Fallback (should not happen)
    return ["black"] * 6

def set_axes_equal(ax):
    """Equal scale on all three axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2

    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2

    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

    ax.set_box_aspect([1, 1, 1])

def visualize(nodes, edges, benchmark, cube_size=0.4, pipe_thickness=0.18, plot=False):
    """Render a pipe diagram (`bgraph_metadata`, `edge_metadata` from
    `export.bgraph.build_pipe_diagram`) with Matplotlib; with `plot=True`
    the figure is saved to `result/visualization/<benchmark>.png`."""
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection="3d")

    # draw nodes
    for node in tqdm(nodes.values()):
        face_colors = get_node_face_colors(node)
        draw_node(ax, node["position"], cube_size, face_colors)

    # draw edges
    for (n1, n2), edge_info in tqdm(edges.items()):
        draw_connected_edge(
            ax,
            nodes[n1]["position"],
            nodes[n2]["position"],
            nodes[n1],
            nodes[n2],
            cube_size,
        )

    set_axes_equal(ax)
    if plot==True:
        plt.savefig(
            f"result/visualization/{benchmark}",
            dpi=300,              # high quality
            bbox_inches="tight",  # remove extra white margins
            pad_inches=0.02
            )
    plt.show()

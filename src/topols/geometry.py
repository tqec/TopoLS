# ---------------------------------------------------------------------------
# Utility functions for 3D point operations
# ---------------------------------------------------------------------------

def neg(p):
    """
    Returns the negation of a 3D point or vector.
    """
    return (-p[0], -p[1], -p[2])

def add(p, d):
    """
    Computes the vector addition of two 3D points or vectors.
    """
    return (p[0]+d[0], p[1]+d[1], p[2]+d[2])

def manhattan(a, b):
    """
    Computes the Manhattan (L1) distance between two 3D points.
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def vector(p, q):
    """
    Returns the displacement vector from point p to point q in 3D space.
    """
    return (q[0]-p[0], q[1]-p[1], q[2]-p[2])

def bounding_box(points, paths, x_max_floor, x_min_floor, y_max_floor, y_min_floor, min_z, z_length):
    """
    Computes the volume of the bounding box enclosing all points and paths under given spatial constraints.
    """
    position_points = [value for _, value in points.items()]
    path_points = [pt for path in paths for pt in path]
    all_points = position_points + path_points
    _, _, zs = zip(*all_points)
    vol = (x_max_floor - x_min_floor + 1) * (y_max_floor - y_min_floor + 1) * (max(zs) - min_z + z_length)
    return vol

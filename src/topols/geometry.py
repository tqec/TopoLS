"""Small helpers on integer 3D points and the bounding-box volume of an
embedding.
"""

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
    """Space-time volume of a partial embedding.

    Args:
        points: `{node: (x, y, z)}`; paths: iterable of paths (cell tuples).
        x_max_floor, x_min_floor, y_max_floor, y_min_floor: footprint limits;
            the x/y extent is fixed by these, not by the points.
        min_z: z of the layer's floor.
        z_length: height already accumulated by earlier layers.

    Returns:
        `x_extent * y_extent * (max_z - min_z + z_length)` where `max_z` is the
        highest cell in `points` or `paths`.
    """
    """
    Computes the volume of the bounding box enclosing all points and paths under given spatial constraints.
    """
    # x/y extent comes from the floor parameters; only z is measured from
    # the points and paths. Called on every EmbeddingState construction.
    max_z = max(pt[2] for pt in points.values())
    for path in paths:
        for pt in path:
            if pt[2] > max_z:
                max_z = pt[2]
    return (x_max_floor - x_min_floor + 1) * (y_max_floor - y_min_floor + 1) * (max_z - min_z + z_length)

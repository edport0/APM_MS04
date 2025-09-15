import numpy as np
from dataclasses import dataclass

@dataclass
class SimpleBoundaryMesh:
    points: np.ndarray        # (N, 2)
    segments: np.ndarray      # (M, 2) index pairs
    extremities: np.ndarray   # (M, 2, 2) coordinates [[p0, p1], ...]

def build_mesh_from_points(points: np.ndarray, *, closed: bool = True) -> SimpleBoundaryMesh:
    """
    Build a simple boundary mesh from ordered 2D points.
    - points: array of shape (N, 2), ordered along the boundary (CCW or CW).
    - closed: if True, connect last point back to first.

    Returns:
      SimpleBoundaryMesh with points, segments (indices), and extremities (coords).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    n = pts.shape[0]
    if closed and n < 3:
        raise ValueError("closed=True requires at least 3 points")
    if not closed and n < 2:
        raise ValueError("closed=False requires at least 2 points")

    i0 = np.arange(n)
    i1 = (i0 + 1) % n if closed else (i0 + 1)
    segments = np.stack([i0[:(n if closed else n-1)],
                         i1[:(n if closed else n-1)]], axis=1)

    extremities = np.stack([pts[segments[:, 0]], pts[segments[:, 1]]], axis=1)  # (M, 2, 2)

    return SimpleBoundaryMesh(points=pts, segments=segments, extremities=extremities)


# --- Example: Disk mesh ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define circle points
    R = 1.0
    n_points = 40
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    pts = np.column_stack((R*np.cos(theta), R*np.sin(theta)))

    # Build mesh
    mesh = build_mesh_from_points(pts, closed=True)

    # Print some info
    print("Points:\n", mesh.points[:5], "...")
    print("Segments (indices):\n", mesh.segments[:5], "...")
    print("Extremities (coords):\n", mesh.extremities[:2], "...")

    # Plot
    cyc = np.vstack([mesh.points, mesh.points[0]])
    plt.plot(cyc[:,0], cyc[:,1], 'o-', label="boundary points")
    for seg in mesh.extremities:
        x, y = seg[:,0], seg[:,1]
        plt.plot(x, y, 'k-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(); plt.title("Disk boundary mesh")
    plt.show()

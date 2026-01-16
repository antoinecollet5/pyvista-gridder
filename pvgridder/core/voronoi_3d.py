import numpy as np
import scipy as sp
import pyvista as pv
import numba
from typing import List, Tuple, Optional
from scipy.spatial import HalfspaceIntersection
from concurrent.futures import ThreadPoolExecutor
import shapely

import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]
NDArrayBool = npt.NDArray[np.bool_]


def get_mask_points_in_domain(
    points: NDArrayFloat,
    domain: pv.PolyData,
    tolerance: float = 0.001,
    inside_out: bool = False,
) -> NDArrayBool:
    """
    Mark points as to whether they are inside a closed surface.

    Parameters
    ----------
    points: NDArrayFloat
        Voronoi sites. Numpy array with shape (N, 3).
    domain : pv.Polydata
        Closed surface in which points must lie.
    tolerancefloat, default: 0.001
        The tolerance on the intersection. The tolerance is expressed as a fraction of the bounding box of the enclosing surface.
    inside_outbool, default: False
        By default, points inside the surface are marked inside or sent to the output. If inside_out is True, then the points outside the surface are marked inside.

    Returns
    -------
    NDArrayInt
        Mask array.

    """
    # mask for points inside the given polydata
    inside_pts_mask = np.ones(shape=np.shape(points)[0], dtype=bool)
    # initialize the valid points
    valid_points = np.copy(points)
    # initialize counters old/new
    n_valid_points_old = np.size(valid_points) + 1
    n_valid_points_new = n_valid_points_old - 1
    # repeat the filtering operation as long as there is a change (this is because
    # ``select_enclosed_points`` is not exact
    while n_valid_points_new < n_valid_points_old:
        # update the number of valid points
        n_valid_points_old = np.size(valid_points)
        # Update the mask
        inside_pts_mask[inside_pts_mask] = (
            pv.PolyData(valid_points)
            .select_enclosed_points(
                domain, check_surface=False, inside_out=inside_out, tolerance=tolerance
            )["SelectedPoints"]
            .view(bool)
        )
        # filter points
        valid_points = points[inside_pts_mask]
        # update comptor
        n_valid_points_new = np.size(valid_points)
    return inside_pts_mask


def _make_sanity_checks(
    points: NDArrayFloat,
    domain: pv.PolyData,
) -> None:
    """
    Check that points have shape (N, 3) and that domain is manifold.

    Parameters
    ----------
    points: NDArrayFloat
        Voronoi sites. Numpy array with shape (N, 3).
    domain : pv.Polydata
        Closed surface in which the tesselation must occur.

    Raises
    ------
    ValueError
        If the point cloud shape is incorrect. If the domain is not manifold or
        if all points are not enclosed in the domain.
    """

    # Step 1: Sanity checks - check the dimension of the point cloud
    _shape = np.shape(points)
    if len(_shape) != 2:
        raise ValueError(
            f"Points have shape {_shape} while the it should have shape (N, 3)!"
        )
    # Number of voronoi cells and dimension
    _, dim = _shape
    # Check that points are 3D
    if dim != 3:
        raise ValueError(
            f"Points are {np.size(points[0])}D while the "
            "code supports 3D voronoi diagram."
        )

    # Step 2: Sanity checks - check the domain
    if not domain.is_manifold:
        raise ValueError("`domain` is not manifold")

    # Step 3: Sanity checks - check that all voronoi points are in the domain.
    # Create a PyVista point cloud
    if not np.all(
        pv.PolyData(points).select_enclosed_points(domain)["SelectedPoints"].view(bool)
    ):
        raise ValueError(
            "All points should be included in the domain!"
            "Use `pv.PolyData(points).select_enclosed_points(domain)['SelectedPoints']"
            ".view(bool)` to check which points are lying outside the domain!"
        )


def get_bounding_cube_coords(domain: pv.PolyData, margin: float = 0.1) -> NDArrayFloat:
    """
    Return the coordinates of the bounding cube with shape (3, 2).

    ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    Parameters
    ----------
    domain: pv.PolyData
        Closed surface in which the tesselation must occur.
    margin: float = 0.1
        Margin factor by which the bounding cube dimensions are increased.

    """
    coords = np.reshape(domain.bounds, (2, -1), order="F").T
    if margin != 0.0:
        ext = np.diff(coords, axis=1).ravel() / 2.0 * (1 + margin)
        mean = np.mean(coords, axis=1)
        coords[:, 0] = mean - ext
        coords[:, 1] = mean + ext
    return coords


def get_bounding_cube(domain: pv.PolyData, margin: float = 0.1) -> pv.Cube:
    """
    Return a bounding cube.

    Parameters
    ----------
    domain: pv.PolyData
        Closed surface in which the tesselation must occur.
    margin: float = 0.1
        Margin factor by which the bounding cube dimensions are increased.

    """
    cube_coords = get_bounding_cube_coords(domain, margin=margin)
    x_length, y_length, z_length = np.diff(cube_coords, axis=1)
    return pv.Cube(
        center=np.mean((cube_coords), axis=1),
        x_length=x_length,
        y_length=y_length,
        z_length=z_length,
    )


@numba.njit(parallel=True, cache=True)
def _build_all_halfspaces_parallel_jit(
    points,
    neighbors_flat: NDArrayInt,
    neighbors_offsets: NDArrayInt,
    bbox: NDArrayFloat,
):
    """
    Construct half-space representations for Voronoi cells in parallel.

    For each input point, this function builds the set of half-spaces
    defining its Voronoi cell. Each half-space corresponds either to
    a bisector plane between the point and one of its neighbors, or to
    a bounding plane of the global domain bounding cube.

    The computation is parallelized over points using Numba and expects
    a flattened adjacency representation for efficient JIT compilation.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Coordinates of the Voronoi sites.
    neighbors_flat : ndarray of shape (M,), dtype int64
        Flattened array of neighbor indices for all points.
        Neighbors for point ``i`` are stored in the slice
        ``neighbors_flat[neighbors_offsets[i]:neighbors_offsets[i+1]]``.
    neighbors_offsets : ndarray of shape (N + 1,), dtype int64
        Offset array delimiting the neighbors of each point in
        ``neighbors_flat``.
    bbox : ndarray of shape (6, 4)
        Half-space representation of the bounding domain. Each row
        represents a plane of the form::

            a * x + b * y + c * z + d <= 0

        where ``(a, b, c)`` is the outward normal and ``d`` is the offset.

    Returns
    -------
    halfspaces_list : list of ndarray
        List of length ``N``, N being the number of voronoi sites.
        Each entry is an array of shape ``(num_neighbors_i + 6, 4)``
        containing the half-space coefficients for the Voronoi cell of the
        corresponding point.

        Neighbor half-spaces are listed first, followed by the bounding
        box planes (6 planes).

    Notes
    -----
    * This function is compiled with ``numba.njit(parallel=True)`` and
      is intended for high-performance batch computation.
    * All returned half-spaces are oriented such that the corresponding
      point satisfies the inequality.
    * The returned Python list is allocated inside the JIT-compiled
      function; this is supported by Numba but limits downstream
      vectorization.

    See Also
    --------
    scipy.spatial.HalfspaceIntersection :
        Consumes half-space representations to compute polyhedral cells.
    """
    n_points = points.shape[0]
    halfspaces_list = [np.empty((0, 4), dtype=np.float64) for _ in range(n_points)]
    # halfspaces_list = np.empty((n_points, 0, 4), dtype=np.float64)

    for i in numba.prange(n_points):
        neighbors = neighbors_flat[neighbors_offsets[i] : neighbors_offsets[i + 1]]
        p = points[i]

        n_neighbors = len(neighbors)
        n_total = n_neighbors + bbox.shape[0]
        hs = np.empty((n_total, 4), dtype=np.float64)

        # Neighbor planes
        for j in range(n_neighbors):
            q = points[neighbors[j]]

            n = q - p
            norm = np.sqrt(np.sum(n**2))
            n /= norm

            mid = 0.5 * (p + q)
            b = -(n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2])

            if n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + b > 0:
                n = -n
                b = -b

            hs[j, 0:3] = n
            hs[j, 3] = b

        for k in range(bbox.shape[0]):
            hs[n_neighbors + k, :] = bbox[k, :]

        halfspaces_list[i] = hs

    return halfspaces_list


def _build_halfspaces_per_cell(
    points: NDArrayFloat,
    domain: pv.PolyData,
) -> List[NDArrayFloat]:
    """
    Build half-space representations of 3D Voronoi cells within a bounded domain.

    This function computes the half-space inequalities defining the Voronoi
    cell of each input point, clipped to a closed 3D domain. The workflow is:

    1. Extract an axis-aligned bounding box enclosing the domain.
    2. Compute a 3D Delaunay tetrahedralization of the input points.
    3. Derive point adjacency from tetrahedral edges.
    4. Construct Voronoi half-spaces in parallel using neighbor bisector planes
       and domain bounding planes.

    The resulting half-spaces can be passed directly to
    ``scipy.spatial.HalfspaceIntersection`` for polyhedral reconstruction.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Coordinates of the Voronoi sites.
    domain : pyvista.PolyData
        Closed surface defining the spatial domain in which the Voronoi
        tessellation is constrained.

    Returns
    -------
    halfspaces_per_cell : list of ndarray
        List of length ``N``. Each element is an array of shape
        ``(num_neighbors_i + 6, 4)`` containing the half-space coefficients
        ``(a, b, c, d)`` such that::

            a * x + b * y + c * z + d <= 0

        defines the clipped Voronoi cell for the corresponding site.

    Notes
    -----
    * The bounding domain is approximated by its axis-aligned bounding box.
    * Neighbor relations are inferred from the Delaunay tetrahedralization,
      which guarantees that all Voronoi-adjacent sites are included.
    * Half-space construction is parallelized via Numba in
      ``build_all_halfspaces_parallel_jit``.

    See Also
    --------
    build_all_halfspaces_parallel_jit :
        Parallel construction of Voronoi half-spaces.
    scipy.spatial.HalfspaceIntersection :
        Computes polyhedral intersections from half-space representations.
    """
    n_cells = len(points)

    # Step 1: extract the domain bouding cube coordinates and create the cube
    cube_coords = get_bounding_cube_coords(domain, margin=0.1)

    # Step 2: create a plan projection for the cube
    bbox = np.array(
        [
            (-1, 0, 0, cube_coords[0, 0]),
            (1, 0, 0, -cube_coords[0, 1]),
            (0, -1, 0, cube_coords[1, 0]),
            (0, 1, 0, -cube_coords[1, 1]),
            (0, 0, -1, cube_coords[2, 0]),
            (0, 0, 1, -cube_coords[2, 1]),
        ]
    )

    # Step 3: Build 3D Delaunay with PyVista => tetrahedralization
    tetra = pv.PointSet(points).delaunay_3d()

    # Step 4: Extract tetrahedra connectivity
    # VTK stores cells in a flat array:
    # [npts, p0, p1, p2, p3, npts, p0, ...]
    cells = tetra.cells.reshape(-1, 5)[:, 1:]

    # Step 5: Build point adjacency from edges from tetrahedra
    # Each tetrahedron has 6 edges
    # - normalize edge ordering
    # - remove duplicates
    edges = np.unique(
        np.sort(
            np.vstack(
                [
                    cells[:, [0, 1]],
                    cells[:, [0, 2]],
                    cells[:, [0, 3]],
                    cells[:, [1, 2]],
                    cells[:, [1, 3]],
                    cells[:, [2, 3]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )

    # Step 6: Build point adjacency list
    # adj[i] will contain the indices of all points adjacent to point i
    # in the Delaunay tetrahedralization (i.e., Voronoi neighbors).
    adj = [[] for _ in range(n_cells)]

    # Each edge (i, j) implies mutual adjacency
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    # Step 7: Flatten adjacency for Numba-friendly parallel processing
    #
    # We convert the Python list-of-lists structure into two NumPy arrays:
    # - neighbors_flat: A single 1D array containing all neighbor indices.
    # - neighbors_offsets: An offset array such that neighbors of point i are found in:
    # neighbors_flat[neighbors_offsets[i] : neighbors_offsets[i + 1]]
    #
    # This layout avoids Python objects inside the Numba-parallel function.
    # Compute prefix sums to determine offsets
    neighbors_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    for i, neighbors in enumerate(adj):
        neighbors_offsets[i + 1] = neighbors_offsets[i] + len(neighbors)

    # Allocate the flattened neighbor array
    neighbors_flat = np.empty(neighbors_offsets[-1], dtype=np.int64)

    # Fill the flattened array
    k = 0
    for neighbors in adj:
        neighbors_flat[k : k + len(neighbors)] = neighbors
        k += len(neighbors)

    # Step 8: finally build the halfspaces
    return _build_all_halfspaces_parallel_jit(
        points, neighbors_flat, neighbors_offsets, bbox
    )


def _build_3d_cell_local(
    halfspaces: NDArrayFloat,
    p: NDArrayFloat,
) -> NDArrayFloat:
    """
    Compute the vertices of a single 3D Voronoi cell from its half-spaces.

    This function computes the intersection points of a set of half-spaces
    defining a convex polyhedron using
    ``scipy.spatial.HalfspaceIntersection``.

    Parameters
    ----------
    halfspaces_i : ndarray of shape (M, 4)
        Half-space representation of a single Voronoi cell. Each row
        corresponds to a plane of the form::

            a * x + b * y + c * z + d <= 0

    p : ndarray of shape (3,)
        A point strictly inside the feasible region defined by
        ``halfspaces_i``. This point is required by
        ``HalfspaceIntersection`` as a starting point for the intersection
        computation.

    Returns
    -------
    vertices : ndarray of shape (V, 3)
        Coordinates of the vertices of the convex polyhedron corresponding
        to the Voronoi cell.

    Notes
    -----
    * The input point ``p`` must lie strictly inside all half-spaces;
      otherwise, the intersection computation will fail.
    * This function operates on a single cell and is intended to be used
      either serially or within a parallel execution context.

    See Also
    --------
    scipy.spatial.HalfspaceIntersection :
        Computes intersections of half-spaces defining a convex polyhedron.
    """
    hs = HalfspaceIntersection(halfspaces, p)
    return hs.intersections


def _compute_vertices_per_cell(
    points: NDArrayFloat,
    halfspaces: List[NDArrayFloat],
    max_workers: int = -1,
    eps: float = 1e-5,
) -> List[NDArrayFloat]:
    """
    Compute Voronoi cell vertices from half-space representations.

    This function computes the vertices of each Voronoi cell defined by a
    list of half-space arrays. For numerical robustness, the site
    coordinates are slightly perturbed to ensure they lie strictly inside
    their corresponding half-space intersections.

    Depending on the number of cells, the computation is performed either
    serially or in parallel using a thread pool.

    Parameters
    ----------
    points: NDArrayFloat
        Coordinates of the Voronoi sites: ndarray of shape (N, 3).

    halfspaces : list of ndarray
        List of length ``N`` containing half-space representations of the
        Voronoi cells. Each entry is an array of shape ``(M_i, 4)`` defining
        planes of the form::

            a * x + b * y + c * z + d <= 0

    max_workers : int, optional
        Maximum number of worker threads used for parallel computation.
        If set to ``-1`` (default), the executor uses the system default.
        Parallel execution is automatically disabled for small problems (number of
        voronoi sites below 500).

    eps : float, optional
        Small positive offset added to each site coordinate to ensure that
        the initial point lies strictly inside the feasible region.
        Default is ``1e-5``.

    Returns
    -------
    vertices_per_cell : list of ndarray
        List of length ``N``. Each element is an array of shape ``(V_i, 3)``
        containing the vertices of the corresponding Voronoi cell.

    Notes
    -----
    * For fewer than ~500 cells, serial execution is preferred, as thread
      management overhead outweighs parallel speedup.
    * This function relies on ``scipy.spatial.HalfspaceIntersection``, which
      assumes convex, bounded regions.
    * Thread-based parallelism is used instead of multiprocessing to avoid
      excessive memory duplication.

    See Also
    --------
    _build_3d_cell_local :
        Computes vertices for a single Voronoi cell.
    scipy.spatial.HalfspaceIntersection :
        Half-space intersection algorithm for convex polyhedra.
    """
    # Number of voronoi sites/cells
    n_cells = len(halfspaces)

    # if the number of external cells is below 500, no need for multi-processing (slower)
    if n_cells < 500:
        _max_workers: int = 1
    else:
        _max_workers = max_workers

    # Create the halfspaces intersections
    vertices_per_cell: List[NDArrayFloat] = []

    # Single worker (no multi-processing)
    if _max_workers == 1:
        for cell_id in range(n_cells):
            vertices_per_cell.append(
                _build_3d_cell_local(halfspaces[cell_id], points[cell_id] + eps)
            )
    # Multi-processing enabled
    else:
        with ThreadPoolExecutor(max_workers=_max_workers) as executor:
            vertices_per_cell = list(
                executor.map(
                    _build_3d_cell_local,
                    halfspaces,
                    points + eps,
                )
            )
    return vertices_per_cell


@numba.njit(cache=True, fastmath=True)
def _make_plane_basis(n: NDArrayFloat) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Construct an orthonormal basis for a plane from its normal vector.

    Given a normal vector ``n``, this function computes two orthonormal
    vectors ``u`` and ``v`` such that:

    * ``u`` and ``v`` lie in the plane orthogonal to ``n``
    * ``u ⟂ v``
    * ``n ⟂ u`` and ``n ⟂ v``

    The basis construction is numerically stable and avoids choosing
    vectors nearly parallel to the input normal.

    Parameters
    ----------
    n : ndarray of shape (3,)
        Normal vector of the plane. The vector does not need to be
        normalized.

    Returns
    -------
    u : ndarray of shape (3,)
        First unit vector spanning the plane orthogonal to ``n``.

    v : ndarray of shape (3,)
        Second unit vector spanning the plane orthogonal to ``n``,
        defined as ``v = n × u``.

    Notes
    -----
    * The returned vectors ``u`` and ``v`` form a right-handed coordinate
      system with ``n``.
    * This function is compiled with ``numba.njit`` and is intended for
      use in performance-critical inner loops.
    * No explicit check is performed for ``n = 0``; the input normal must
      be non-zero.

    See Also
    --------
    numpy.cross :
        Cross product of two vectors.
    """
    # Choose a vector not parallel to n
    if abs(n[0]) > abs(n[2]):
        a = np.array([-n[1], n[0], 0.0])
    else:
        a = np.array([0.0, -n[2], n[1]])

    # Normalize u
    u = a / np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    # v = n × u
    v = np.array(
        [
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0],
        ]
    )

    return u, v


@numba.njit(cache=True, fastmath=True)
def _get_face_from_halfsace(
    vertices: np.ndarray,
    num_vertices: int,
    halfspace: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Extract and order the polygonal face induced by a half-space.

    Given a set of vertices of a convex polyhedron and a half-space
    defining one of its supporting planes, this function identifies
    the vertices lying on the plane and returns them ordered as a
    planar polygon.

    The ordering is performed counterclockwise in the plane defined
    by the half-space normal.

    Parameters
    ----------
    vertices : ndarray of shape (V, 3)
        Coordinates of the polyhedron vertices.

    num_vertices : int
        Number of valid vertices in ``vertices``. This is provided
        explicitly to avoid repeated shape lookups in performance-
        critical code paths.

    halfspace : ndarray of shape (4,)
        Half-space coefficients ``(a, b, c, d)`` defining the plane::

            a * x + b * y + c * z + d = 0

        The plane normal is assumed to point outward.

    tol : float, optional
        Numerical tolerance used to determine whether a vertex lies
        on the plane. Default is ``1e-8``.

    Returns
    -------
    face_indices : ndarray of shape (F,), dtype int64
        Indices of vertices forming the polygonal face induced by the
        half-space, ordered counterclockwise in the plane.

        If fewer than three vertices lie on the plane, ``None`` is
        returned.

    Notes
    -----
    * A valid face must have at least three coplanar vertices.
    * For exactly three vertices, no reordering is necessary and the
      indices are returned directly.
    * For more than three vertices, the face is ordered by projecting
      the vertices onto a local 2D basis constructed from the plane
      normal and sorting by polar angle.
    * This function assumes the input polyhedron is convex.

    See Also
    --------
    _make_plane_basis :
        Constructs an orthonormal basis for a plane.
    numpy.argsort :
        Indirect sort used to order vertices by angle.

    """
    n = halfspace[:3]
    d = halfspace[3]

    # Collect indices of vertices lying on the plane
    tmp = np.empty(num_vertices, dtype=np.int64)
    count = 0

    for j in range(num_vertices):
        dist = abs(
            vertices[j, 0] * n[0] + vertices[j, 1] * n[1] + vertices[j, 2] * n[2] + d
        )
        if dist < tol:
            tmp[count] = j
            count += 1

    # return something empty => the slicing is required for numba to compile
    if count < 3:
        return tmp[:0]

    face_idx = tmp[:count]

    # If there are only three points => triangle face and the order dos not matter
    if count == 3:
        return np.asarray(face_idx)

    # ------------------------------------------------------------------
    # Order coplanar vertices
    # ------------------------------------------------------------------

    # Compute centroid
    center = np.zeros(3)
    for k in range(count):
        center += vertices[face_idx[k]]
    center /= count

    # Plane basis
    u, v = _make_plane_basis(n)

    # Compute angles
    angles = np.empty(count)
    for k in range(count):
        rel = vertices[face_idx[k]] - center
        x = rel[0] * u[0] + rel[1] * u[1] + rel[2] * u[2]
        y = rel[0] * v[0] + rel[1] * v[1] + rel[2] * v[2]
        angles[k] = np.arctan2(y, x)

    # Sort by angle
    order = np.argsort(angles)
    ordered = np.empty(count, dtype=np.int64)
    for k in range(count):
        ordered[k] = face_idx[order[k]]

    return ordered


@numba.njit(cache=True, fastmath=True)
def _extract_faces_from_halfspaces(
    halfspaces: NDArrayFloat, vertices: NDArrayFloat, tol: float = 1e-8
) -> Tuple[NDArrayFloat, numba.typed.List[NDArrayInt]]:
    """
    Extract polygonal faces from a convex polyhedron defined by halfspaces.

    Each face corresponds to one plane and is returned as an ordered list
    of vertex indices lying on that plane.

    Parameters
    ----------
    halfspaces : (M, 4) ndarray
        Plane equations in the form [a, b, c, d] representing
        ax + by + cz + d = 0.
    vertices : (N, 3) ndarray
        Vertices of the polyhedron.
    tol : float, optional
        Tolerance for point-to-plane distance.

    Returns
    -------
    faces : list of ndarray
        Each entry is an array of vertex indices forming a face.
        Vertices are ordered counter-clockwise in the plane.
    """
    # Initiate an empty
    faces = numba.typed.List()
    # Number of of vertices defining the voronoi cells.
    num_vertices = vertices.shape[0]
    # build the faces
    for i in range(halfspaces.shape[0]):
        res: NDArrayInt = _get_face_from_halfsace(
            vertices, num_vertices, halfspaces[i], tol=tol
        )
        if np.size(res) != 0:
            faces.append(res)

    return vertices, faces


# Project 3D points to 2D coordinates on the plane
def project_to_plane(points, plane_normal, plane_point):
    # Build plane basis
    if abs(plane_normal[0]) < 0.9:
        u = np.cross(plane_normal, [1, 0, 0])
    else:
        u = np.cross(plane_normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)
    # 3D -> 2D
    xy = np.empty((points.shape[0], 2))
    for i in range(points.shape[0]):
        rel = points[i] - plane_point
        xy[i, 0] = np.dot(rel, u)
        xy[i, 1] = np.dot(rel, v)
    return xy, u, v


# @numba.njit(cache=True, fastmath=True)
def extract_faces_from_halfspaces_with_domain(
    halfspaces: np.ndarray, vertices: np.ndarray, domain: pv.PolyData, tol: float = 1e-8
) -> Tuple[numba.typed.List[np.ndarray], np.ndarray]:
    """
    Extract polygonal faces from a convex polyhedron defined by halfspaces.

    Each face corresponds to one plane and is returned as an ordered list
    of vertex indices lying on that plane.

    Parameters
    ----------
    halfspaces : (M, 4) ndarray
        Plane equations in the form [a, b, c, d] representing
        ax + by + cz + d = 0.
    vertices : (N, 3) ndarray
        Vertices of the polyhedron.
    tol : float, optional
        Tolerance for point-to-plane distance.

    Returns
    -------
    faces : list of ndarray
        Each entry is an array of vertex indices forming a face.
        Vertices are ordered counter-clockwise in the plane.
    """
    faces = numba.typed.List()
    num_vertices = vertices.shape[0]
    num_halfspaces = halfspaces.shape[0]

    # Step 1: find what vertices are on the bounding cube (vertices to remove)
    boundary_vertices_mask = np.zeros(num_vertices, dtype=bool)
    for i in range(6):
        face = _get_face_from_halfsace(
            vertices, num_vertices, halfspaces[num_halfspaces - 1 - i]
        )
        if face is not None:
            # mark these vertices are boundaries
            boundary_vertices_mask[face] = True

    boundary_vertices = np.arange(num_vertices)[boundary_vertices_mask]

    # If none on the bounding cube, call the classic method to build the faces.
    if np.size(boundary_vertices) == 0:
        return _extract_faces_from_halfspaces(halfspaces, vertices, tol)

    # remove useless vertices
    new_vertices = [vertices[boundary_vertices]]
    kept_halfspaces_idx = []

    # Step 2: check what face contains these points => faces that intersect the bounding
    # cube and for which the plant must be projected on the surface => new points are
    # obtained.
    for i in range(num_halfspaces):
        is_cube_intersecting: bool = False
        face = _get_face_from_halfsace(vertices, num_vertices, halfspaces[i])
        # Handle the case with no face for that halfspace
        if face is None:
            continue
        # Otherwise iterate the faces and check if it intersects
        for v in face:
            if v in boundary_vertices:
                is_cube_intersecting = True
                break

        # For each face, store the halfspace so that the face can be reconstructed later on
        kept_halfspaces_idx.append(i)

        # If needed, project the surface
        if is_cube_intersecting:
            # Normal vector
            normal = halfspaces[i][:3]
            # norm = np.linalg.norm(normal)
            # normal /= norm

            # Point on plane
            d = halfspaces[i][3]
            point_on_plane = -d * normal  # satisfies ax+by+cz+d=0

            # Slice the domain with the plane
            slice_poly = domain.slice(normal=normal, origin=point_on_plane)

            print(normal)

            # Project on the plane to perform 2D intersection using shapely
            # Slice polygon and face polygon
            slice_xy, u, v = project_to_plane(slice_poly.points, normal, point_on_plane)
            face_xy, _, _ = project_to_plane(vertices, normal, point_on_plane)

            # TODO: shapely cutting
            # Shapely intersection
            print(slice_xy, face_xy)
            clipped_2d = shapely.Polygon(slice_xy).intersection(
                shapely.Polygon(face_xy)
            )
            # shapely.Polygon(vertices[face]).intersection(shapely.Polygon(slice_poly.points))

            # Back to 3D
            # Map back to 3D
            clipped_3d = np.array(
                [
                    point_on_plane + x * u + y * v
                    for x, y in np.array(clipped_2d.exterior.coords)
                ]
            )
            new_vertices.append(clipped_3d)

    # Step 3: compute the convex hull for the voronoi cell.
    # Make vertices unique
    vertices = np.unique(np.vstack(new_vertices), axis=1)
    num_vertices = np.shape(vertices)[0]

    # compute the convexhull
    hull = sp.spatial.ConvexHull(vertices)
    # Get the faces
    faces = hull.simplices

    # Step 4: merge coplanar faces (using halfspace).
    for i in kept_halfspaces_idx:
        face = _get_face_from_halfsace(vertices, num_vertices, halfspaces[i])

    print(faces)

    return vertices, faces


def _build_unstructured_grid(
    vertices_per_cell: List[NDArrayFloat],
    halfspaces_per_cell: List[NDArrayFloat],
    domain: pv.PolyData,
    is_clip_to_domain: bool = False,
    tolerance: float = 1e-10,
) -> pv.UnstructuredGrid:
    """
    Build an unstructured grid from the vertices and halfspaces of the voronoi cells.

    Parameters
    ----------
    vertices_per_cell : List[NDArrayFloat]
        List of vertices arrays with shape (nv_i, 3) given for each voronoi cell,
        nv_i being the number of vertices for the ith cell.
    halfspaces_per_cell : List[NDArrayFloat]
        List of halfspaces arrays with shape (nv_i, 4) given for each voronoi cell,
        nv_i being the number of halfspace for the ith cell.
    domain : pv.PolyData
        Voronoi domain.
    is_clip_to_domain: bool
        Whether the voronoi cells must be clipped to the input domain closed surface.
        If False, then the bounding cube is used instead of the domain.
        The default is False.
    tolerance : float, optional
        Tolerance for merging duplicate points and remove unused points in
        the output UnstructuredGrid., by default 1e-10.

    Returns
    -------
    pv.UnstructuredGrid

    """
    # Build unstructured grid
    total_n_pts: int = 0
    n_cells: int = 0
    cells_def = []

    for i, (_vertices, _halfspaces) in enumerate(
        zip(vertices_per_cell, halfspaces_per_cell)
    ):
        # Build the faces
        if is_clip_to_domain:
            _vertices, faces = extract_faces_from_halfspaces_with_domain(
                _halfspaces, _vertices, domain
            )
        else:
            _vertices, faces = _extract_faces_from_halfspaces(
                _halfspaces,
                _vertices,
            )
            # Update the vertices
            vertices_per_cell[i] = _vertices
        # updated number of vertices for the voronoi cell
        n_pts = len(_vertices)
        # Add the number of faces
        polyhedron_connectivity = [len(faces)]
        # iterate faces and update the connectivity for each one
        for f in faces:
            polyhedron_connectivity += [len(f)] + list(np.array(f) + total_n_pts)
        cells_def += [len(polyhedron_connectivity), *polyhedron_connectivity]
        # update the total number of points
        total_n_pts += n_pts
        n_cells += 1

    return pv.UnstructuredGrid(
        np.array(cells_def),
        [pv.CellType.POLYHEDRON] * n_cells,
        np.vstack(vertices_per_cell),
    ).clean(tolerance=tolerance)


def voronoi_finite_cells(
    points: NDArrayFloat,
    domain: pv.PolyData,
    eps: float = 1e-5,
    max_workers: int = 1,
    is_clip_to_domain: bool = False,
):
    """
    Generate the voronoi 3D tesselation from the input point within the domain bounds.

    Parameters
    ----------
    points: NDArrayFloat
        Voronoi sites. Numpy array with shape (N, 3).
    domain : pv.Polydata
        Closed surface in which the tesselation must occur.
    eps : float, optional
        Small positive offset added to each site coordinate to ensure that
        the initial point lies strictly inside the feasible region.
        Default is ``1e-5``.
    max_workers : int, optional
        Maximum number of worker threads used for parallel computation.
        If set to ``-1`` (default), the executor uses the system default.
        Parallel execution is automatically disabled for small problems (number of
        voronoi sites below 500).
    is_clip_to_domain: bool
        Whether the voronoi cells must be clipped to the input domain closed surface.
        If False, then the bounding cube is used instead of the domain.
        The default is False.

    Returns
    -------
    pv.UnstructuredGrid
        The finite voronoi cells as an unstructured grid.

    """
    # Step 1: sanity checks
    _make_sanity_checks(points, domain)

    # Step 2: create the halfspaces for each voronoi cell (all plans defining the cells)
    # The arrays have shape (nb neigbors + 6, 4)
    halfspaces_per_cell: List[NDArrayFloat] = _build_halfspaces_per_cell(points, domain)

    # Step 3: get the cell vertices from the halfspaces intersection
    vertices_per_cell: List[NDArrayFloat] = _compute_vertices_per_cell(
        points, halfspaces_per_cell, max_workers=max_workers, eps=eps
    )

    # TODO: wrap in a function
    # Flatten vertices before checking which are enclosed in the domain
    # so as to beneficiate from vectorization
    vertices_in_domain = get_mask_points_in_domain(np.vstack(vertices_per_cell), domain)

    # Iterate the cells
    boundary_cell_ids = []
    # start index for the current cell in the flatten vertices
    start_idx = 0
    # Iterate all cells and associated vertices to check whether the cell lies fully in
    # the domain or if it needs to be cropped.
    for i, cell_vertices in enumerate(vertices_per_cell):
        # update the end index for the current cell in the flatten vertices
        end_idx = start_idx + len(cell_vertices)
        cell_vertices_in_domain = vertices_in_domain[start_idx:end_idx]
        # First case: all vertices are in the domain => keep the cell as it is
        if np.all(cell_vertices_in_domain):
            pass
        # Case two: some vertices are outside of the domain => it the a boundary cell
        # and it needs to be clipped.
        else:
            boundary_cell_ids.append(i)
        # Note that the case with all vertices outside can not append normally so it
        # is not tested.

        # update the end index for the current cell in the flatten vertices
        start_idx = end_idx

    n_cells = len(points)
    n_ext_cells = len(boundary_cell_ids)
    print(
        f"There are {n_ext_cells} boundary cells over "
        f"{n_cells} ({n_ext_cells / n_cells * 100.0:.2f}%)"
    )

    return _build_unstructured_grid(
        vertices_per_cell,
        halfspaces_per_cell,
        domain,
        is_clip_to_domain,
        tolerance=1e-10,
    )

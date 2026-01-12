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


def get_bounding_cube_coords(domain: pv.PolyData) -> NDArrayFloat:
    """
    Return the coordinates of the bounding cube with shape (3, 2).

    ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    return np.reshape(domain.bounds, (2, -1), order="F").T


def get_bounding_cube(domain: pv.PolyData) -> pv.Cube:
    """Return a bounding cube."""
    cube_coords = get_bounding_cube_coords(domain)
    x_length, y_length, z_length = np.diff(cube_coords, axis=1)
    return pv.Cube(
        center=np.mean((cube_coords), axis=1),
        x_length=x_length,
        y_length=y_length,
        z_length=z_length,
    )


def get_voronoi_cells_adjacency(vor: sp.spatial.Voronoi) -> List[List[int]]:
    adj = [[] for _ in range(len(vor.points))]
    for p1, p2 in vor.ridge_points:
        adj[p1].append(p2.item())
        adj[p2].append(p1.item())
    return adj


def build_boundary_cells_mask(
    domain: pv.PolyData, vor: sp.spatial.Voronoi
) -> np.typing.NDArray[np.bool]:
    # Initiate the mask
    mask_boundary_cells = np.zeros(np.shape(vor.points)[0], dtype=np.bool)

    # Mask the vertices outside the desired domain
    outside_vertices = ~pv.PolyData(vor.vertices).select_enclosed_points(
        domain, check_surface=True
    )["SelectedPoints"].view(bool)

    # Iterate the regions (for each voronoi cell, list of vertices ids composing the
    # voronoi cell 3D convex hull).
    # If -1 => means it is an "infinite" vertice
    # If outside the domain => must be recomputed
    for cell_id, rid in enumerate(vor.point_region):
        for vid in vor.regions[rid]:
            if vid == -1:
                mask_boundary_cells[cell_id] = 1
                continue
            elif outside_vertices[vid]:
                mask_boundary_cells[cell_id] = 1
                continue

    return mask_boundary_cells


@numba.njit(parallel=True)
def build_all_halfspaces(
    points,
    neighbors_flat: NDArrayInt,
    neighbors_offsets: NDArrayInt,
    bbox: NDArrayFloat,
    ext_mask: Optional[np.typing.NDArray[np.bool]] = None,
):
    """
    Build halfspaces for all points in parallel.

    Parameters
    ----------
    points : (N,3) array of coordinates
    adjacency : list of arrays of neighbor indices
    bbox : (6,4) array of bounding planes

    Returns
    -------
    halfspaces_list : list of arrays, each (num_neighbors+6,4)
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

        # Bounding box: add only for external cells
        if ext_mask is not None:
            if not ext_mask[i]:
                continue
        for k in range(bbox.shape[0]):
            hs[n_neighbors + k, :] = bbox[k, :]

        halfspaces_list[i] = hs

    return halfspaces_list


def build_3d_cell_local(halfspaces_i: NDArrayFloat, p: NDArrayFloat) -> NDArrayFloat:
    hs = HalfspaceIntersection(halfspaces_i, p)
    return hs.intersections


@numba.njit(cache=True, fastmath=True)
def _make_plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a stable orthonormal basis (u, v) for a plane
    given its normal vector n.
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


# Get face


def get_face_from_halfsace(
    vertices: NDArrayFloat, num_vertices: int, plane: NDArrayFloat, tol: float = 1e-8
):
    n = plane[:3]
    d = plane[3]

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

    if count < 3:
        return

    face_idx = tmp[:count]

    if count == 3:
        return face_idx.copy()

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
def extract_faces_from_halfspaces(
    halfspaces: np.ndarray, vertices: np.ndarray, tol: float = 1e-8
) -> numba.typed.List[np.ndarray]:
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

    for i in range(halfspaces.shape[0]):
        plane = halfspaces[i]
        n = plane[:3]
        d = plane[3]

        # Collect indices of vertices lying on the plane
        tmp = np.empty(num_vertices, dtype=np.int64)
        count = 0

        for j in range(num_vertices):
            dist = abs(
                vertices[j, 0] * n[0]
                + vertices[j, 1] * n[1]
                + vertices[j, 2] * n[2]
                + d
            )
            if dist < tol:
                tmp[count] = j
                count += 1

        if count < 3:
            continue

        face_idx = tmp[:count]

        if count == 3:
            faces.append(face_idx.copy())
            continue

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

        faces.append(ordered)

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
        face = get_face_from_halfsace(
            vertices, num_vertices, halfspaces[num_halfspaces - 1 - i]
        )
        if face is not None:
            # mark these vertices are boundaries
            boundary_vertices_mask[face] = True

    boundary_vertices = np.arange(num_vertices)[boundary_vertices_mask]

    # If none on the bounding cube, call the classic method to build the faces.
    if np.size(boundary_vertices) == 0:
        return extract_faces_from_halfspaces(halfspaces, vertices, tol)

    # remove useless vertices
    new_vertices = [vertices[boundary_vertices]]
    kept_halfspaces_idx = []

    # Step 2: check what face contains these points => faces that intersect the bounding
    # cube and for which the plant must be projected on the surface => new points are
    # obtained.
    for i in range(num_halfspaces):
        is_cube_intersecting: bool = False
        face = get_face_from_halfsace(vertices, num_vertices, halfspaces[i])
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
        face = get_face_from_halfsace(vertices, num_vertices, halfspaces[i])

    print(faces)

    return vertices, faces


def voronoi_finite_cells(
    points: NDArrayFloat,
    domain: pv.PolyData,
    eps: float = 1e-6,
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
        Some precision parameter, by default 1e-6. TODO: change that.
    max_workers : int, optional
        Maximum number of workers to parallelize the code (multi-threaded),
        by default 1, i.e., single threaded.
    is_clip_to_domain: bool
        Whether the voronoi cells must be clipped to the input domain closed surface.
        If False, then the bounding cube is used instead of the domain.
        The default is False.

    Returns
    -------
    pv.UnstructuredGrid
        The finite voronoi cells as an unstructured grid.

    """
    # Number of voronoi cells and dimension
    n_cells, dim = np.shape(points)

    # check the dimension
    if dim != 3:
        raise ValueError(
            f"Points are {np.size(points[0])}D while the "
            "code supports 3D voronoi diagram."
        )

    # Step 1: extract the domain bouding cube coordinates and create the cube
    cube_coords = get_bounding_cube_coords(domain)

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
    # Initiate adjacency dict
    adj = [[] for _ in range(tetra.n_points)]
    # Fill the dict
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    # TODO: update this
    neighbors_flat = np.array(
        [n for neighbors in adj for n in neighbors],
        dtype=np.int64,
    )
    neighbors_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    offset = 0
    for i, neighbors in enumerate(adj):
        neighbors_offsets[i] = offset
        offset += len(neighbors)
    neighbors_offsets[-1] = offset

    _halfspaces = build_all_halfspaces(points, neighbors_flat, neighbors_offsets, bbox)

    # if the number of external cells is below 500, no need for multi-processing (slower)
    if n_cells < 500:
        _max_workers: int = 1
    else:
        _max_workers = max_workers

    vertices: List[NDArrayFloat] = []

    # Single worker (no multi-processing)
    if _max_workers == 1:
        for cell_id in range(n_cells):
            vertices.append(
                build_3d_cell_local(_halfspaces[cell_id], points[cell_id] + eps)
            )
    # Multi-processing enabled
    else:
        with ThreadPoolExecutor(max_workers=_max_workers) as executor:
            vertices = list(
                executor.map(
                    build_3d_cell_local,
                    _halfspaces,
                    points + eps,
                )
            )

    # TODO: keep track of the cells/faces at the domain boundary
    # For each face => we build the halfspace + the projection on the domain and we keep
    # the intersection of both.

    # # Step 5: Mask the cells that must be recomputed (with external vertices)
    # ext_cell_mask = build_boundary_cells_mask(domain, vor)
    # ext_cell_mask = ext_cell_mask.copy()

    # # Number of external voronoi cells
    # n_ext_cells = np.count_nonzero(ext_cell_mask)
    # # List of external (boundary) cells
    # ext_cell_ids = np.arange(n_cells, dtype=np.int64)[ext_cell_mask]
    # # List of external (boundary) cells
    # int_cell_ids = np.arange(n_cells, dtype=np.int64)[~ext_cell_mask]

    # print(
    #     f"There are {n_ext_cells} boundary cells over "
    #     f"{n_cells} ({n_ext_cells / n_cells * 100.0:.2f}%)"
    # )

    # Build unstructured grid
    total_n_pts: int = 0
    n_cells: int = 0
    cells_def = []

    for i, vertices_i in enumerate(vertices):
        # Build the faces
        if is_clip_to_domain:
            vertices_i, faces = extract_faces_from_halfspaces_with_domain(
                _halfspaces[i], vertices_i, domain
            )
        else:
            vertices_i, faces = extract_faces_from_halfspaces(
                _halfspaces[i],
                vertices_i,
            )
        # number of vertices for the voronoi cell
        n_pts = len(vertices_i)
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
        np.array(cells_def), [pv.CellType.POLYHEDRON] * n_cells, np.vstack(vertices)
    ).clean(tolerance=1e-10)

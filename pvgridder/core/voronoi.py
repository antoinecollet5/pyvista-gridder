from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, cast, Dict, List

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyrequire import require_package
from scipy.spatial import Voronoi

from ._base import MeshBase, MeshItem
from ._helpers import generate_surface_from_two_lines, resolution_to_perc


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence  # pragma: no cover
    from typing import Optional  # pragma: no cover

    from numpy.typing import ArrayLike  # pragma: no cover
    from typing_extensions import Self  # pragma: no cover


@require_package("shapely>=2.0")
class Polygon:
    """Store a polygon selection used to override group/priority of Voronoi cells."""

    from shapely import Polygon as ShapelyPolygon

    def __init__(
        self,
        poly: ShapelyPolygon,
        interior_poly: ShapelyPolygon,
        group: str,
        interior_group: str,
        priority: int,
        interior_priority: int,
        add_central_cells: bool = False,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        poly : ShapelyPolygon
            Polygon geometry (in the mesh's 2D plane) used to test containment/
            intersection against Voronoi points and cells.
        group : str
            Group name assigned to cells selected by this polygon.
        priority : int
            Priority of this polygon. Only active cells with a (strictly) lower
            priority than *priority* are overridden.
        add_central_cells : bool, default False
            If True, only cells whose Voronoi center lies inside *poly* are
            selected (containment test only). If False, *poly* is treated as a
            thin interface: cells whose Voronoi center lies outside the
            background domain but whose Voronoi polygon intersects *poly* are
            also selected, in addition to those contained by *poly*.

        """
        self.poly: ShapelyPolygon = poly
        self.interior_poly: ShapelyPolygon = interior_poly
        self.group: str = group
        self.interior_group: str = interior_group
        self.priority: int = priority
        self.interior_priority: int = interior_priority
        self.add_central_cells: bool = add_central_cells


@require_package("shapely>=2.0")
class VoronoiMesh2D(MeshBase):
    """
    2D Voronoi mesh class.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Background mesh.
    axis : int, default 2
        Background mesh axis to discard.
    preference : {'cell', 'point'}, default 'cell'
        Determine which data to use for background mesh.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "VoronoiMesh2D"
    __qualname__: str = "pvgridder.VoronoiMesh2D"

    def __init__(
        self,
        mesh: pv.DataSet,
        axis: int = 2,
        preference: Literal["cell", "point"] = "cell",
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a 2D Voronoi mesh."""
        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self._preference = preference
        self._fuse_cells = []
        self.mesh.points[:, self.axis] = 0.0  # type: ignore
        self._polygons: List[Polygon] = []

    def add(
        self,
        mesh_or_points: pv.DataSet | ArrayLike,
        priority: int = 0,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points to Voronoi diagram.

        Parameters
        ----------
        mesh_or_points : pyvista.DataSet | ArrayLike
            Dataset or coordinates of points.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        if not isinstance(mesh_or_points, pv.DataSet):
            mesh_or_points = self._check_point_array(mesh_or_points)
            mesh = pv.PolyData(mesh_or_points)

        else:
            mesh = mesh_or_points.copy()
            mesh = cast(
                Union[pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid], mesh
            )

        mesh.points[:, self.axis] = 0.0  # type: ignore
        item = MeshItem(mesh, group=group, priority=priority)
        self.items.append(item)

        return self

    def add_circle(
        self,
        radius: float,
        constraint_radius: Optional[float] = None,
        resolution: Optional[int | ArrayLike] = None,
        center: Optional[ArrayLike] = None,
        plain: bool = False,
        priority: int = 0,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points from a circle to Voronoi diagram.

        Parameters
        ----------
        radius : scalar
            Circle radius.
        constraint_radius : scalar, optional
            Constraint circle radius. If None, default to 1.5 times *radius*.
        resolution : int | ArrayLike, optional
            Number of subdivisions along the azimuthal axis or relative position of
            subdivisions (in percentage) with respect to the starting angle (0 degree).
        center : ArrayLike, optional
            Center of the circle.
        plain : bool, default False
            If True, fuse all cells within the circle into a single cell.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import Annulus, Circle, MeshMerge

        constraint_radius = (
            constraint_radius if constraint_radius is not None else 1.5 * radius
        )
        dr = constraint_radius - radius

        if dr > 0.0:
            self.add(
                (
                    MeshMerge()
                    .add(Circle(radius - dr, resolution, center=center))
                    .add(Annulus(radius - dr, radius, 1, resolution, center=center))
                    .generate_mesh()
                ),
                priority=priority,
                group=group,
            )
            self.add(
                Annulus(radius, radius + dr, 1, resolution, center=center),
                priority=0,
            )

        elif dr == 0.0:
            self.add(
                Circle(radius, resolution, center=center),
                priority=priority,
                group=group,
            )

        else:
            raise ValueError("invalid constraint radius")

        if plain:
            center = np.zeros(3) if center is None else np.asanyarray(center)
            center = np.insert(center, self.axis, 0.0) if len(center) == 2 else center
            self.fuse_cells.append(
                lambda x: np.linalg.norm(x - center, axis=1) < radius
            )

        return self

    def add_polyline(
        self,
        mesh_or_points: ArrayLike | pv.PolyData,
        width: float,
        preference: Literal["cell", "point"] = "cell",
        padding: Optional[float] = None,
        constraint: int | tuple[int, int] = 1,
        resolution: Optional[int | ArrayLike] = None,
        priority: Optional[int] = None,
        group: Optional[str] = None,
        add_central_cells: bool = True,
    ) -> Self:
        """
        Add points from a polyline to Voronoi diagram.

        Parameters
        ----------
        mesh_or_points : ArrayLike | pyvista.PolyData
            Dataset or coordinates of points.
        width : scalar
            Width of polyline. If *add_central_cell* is False, this is simply
            the distance separating the two constraint lines added on either
            side of the polyline (no cells are generated between them).
        preference : {'cell', 'point'}, default 'cell'
            Determine which coordinates to add:

            - if 'cell', add cell centers of polyline.
            - if 'point', add polyline point coordinates.

        padding : scalar, optional
            Distance between cell centers of first and last points (if
            *preference* = 'cell') and start and end of the polyline, respectively.
            Default is half of *width*.
        constraint : int | tuple[int, int], default 1
            Number of constraint points added at the start and the end of the polyline.
        resolution : int | ArrayLike, optional
            Number of subdivisions along the line or relative position of subdivisions
            (in percentage) with respect to the starting point.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.
        add_central_cells : bool, default True
            If True, generate a finite-width band(s) of **cells** around
            the polyline. Otherwise, treat the polyline as a zero-area
            **interface**: only the two offset lines (`line_a`, `line_b`,
            each offset by half of *width* on either side of the polyline)
            are added to the Voronoi diagram as raw point sets. *width*
            still controls how far apart these two lines are placed, but no
            surface mesh or cells are built between them.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import extract_cells, split_lines

        if not isinstance(mesh_or_points, pv.PolyData):
            mesh = pv.MultipleLines(np.asanyarray(mesh_or_points))

        else:
            mesh = mesh_or_points.copy()

        if isinstance(constraint, int):
            constraint_start = constraint
            constraint_end = constraint

        else:
            constraint_start, constraint_end = constraint

        perc = resolution_to_perc(resolution)
        if add_central_cells:
            perc = [2.0 * perc[0] - perc[1], *perc.tolist(), 2.0 * perc[-1] - perc[-2]]
        else:
            perc = [-1.5, 0.5, 0.5, 2.5]
        # Loop over polylines
        for polyline in split_lines(mesh):
            # Remove axis from points
            points = np.delete(polyline.points, self.axis, axis=1)

            # Calculate new point coordinates if cell centers
            if preference == "cell":
                padding = padding if padding is not None else 0.5 * width
                points = np.vstack(
                    (
                        points[0]
                        + padding
                        * (points[0] - points[1])
                        / np.linalg.norm(points[0] - points[1]),
                        0.5 * (points[:-1] + points[1:]),
                        points[-1]
                        + padding
                        * (points[-1] - points[-2])
                        / np.linalg.norm(points[-1] - points[-2]),
                    )
                )

            # Calculate forward direction vectors
            fdvec = np.diff(points, axis=0)
            fdvec = np.vstack((fdvec, fdvec[-1]))

            # Calculate backward direction vectors
            bdvec = np.diff(points[::-1], axis=0)[::-1]
            bdvec = np.vstack((bdvec[0], bdvec))

            # Append constraint points at the start and at the end of the polyline
            for _ in range(constraint_start):
                points = np.vstack((points[0] - fdvec[0], points))
                fdvec = np.vstack((fdvec[0], fdvec))
                bdvec = np.vstack((bdvec[0], bdvec))

            for _ in range(constraint_end):
                points = np.vstack((points, points[-1] - bdvec[-1]))
                fdvec = np.vstack((fdvec, fdvec[-1]))
                bdvec = np.vstack((bdvec, bdvec[-1]))

            # Calculate normal vectors
            fnorm = np.column_stack((-fdvec[:, 1], fdvec[:, 0]))
            bnorm = np.column_stack((bdvec[:, 1], -bdvec[:, 0]))
            normals = 0.5 * (fnorm + bnorm)
            normals /= np.linalg.norm(normals, axis=1)[:, None]

            # Re-insert axis
            points = np.insert(points, self.axis, 0.0, axis=1)
            normals = np.insert(normals, self.axis, 0.0, axis=1)

            if add_central_cells:
                tvec = 0.5 * width * normals
            else:
                tvec = 0.25 * width * normals

            line_a = points - tvec
            line_b = points + tvec

            # Generate structured grid with constraint cells
            plane = "yz" if self.axis == 0 else "xz" if self.axis == 1 else "xy"
            mesh_ = generate_surface_from_two_lines(line_a, line_b, plane, perc)

            # Identify constraint cells
            shape = [n - 1 for n in mesh_.dimensions if n != 1]
            constraint_ = np.ones(shape, dtype=bool)
            constraint_[constraint_start : shape[0] - constraint_end, 1:-1] = False
            constraint_ = constraint_.ravel(order="F")

            # Add to items
            if add_central_cells:
                item = MeshItem(
                    extract_cells(mesh_, ~constraint_),
                    group=group,
                    priority=priority if priority else 0,
                )
                self.items.append(item)

            item = MeshItem(extract_cells(mesh_, constraint_), group=None, priority=0)
            self.items.append(item)

        return self

    def add_polygon(
        self,
        mesh_or_points: ArrayLike | pv.PolyData | ShapelyPolygon,
        width: float,
        preference: Literal["cell", "point"] = "cell",
        resolution: Optional[int | ArrayLike] = None,
        priority: Optional[int] = None,
        group: Optional[str] = None,
        add_central_cells: bool = True,
        interior_group: Optional[str] = None,
        interior_priority: Optional[int] = None,
    ) -> Self:
        """
        Add points from a closed polygon to Voronoi diagram.

        Same as :meth:`add_polyline`, except the input line is treated as a
        closed ring: direction vectors wrap around the seam instead of being
        extrapolated as they would be for an open line, and the whole ring
        forms a single band item.

        Parameters
        ----------
        mesh_or_points : ArrayLike | pyvista.PolyData
            Dataset or coordinates of points describing the polygon outline.
            Does not need to be pre-closed; it is closed automatically if not.
        width : scalar
            Width of the band traced around the polygon outline. If
            *add_central_cell* is False, this is simply the distance
            separating the two constraint lines added on either side of the
            outline (no wide band, just a thin interface).
        preference : {'cell', 'point'}, default 'cell'
            Determine which coordinates to add:

            - if 'cell', use the midpoint of each outline edge.
            - if 'point', use the outline vertices directly.

        padding : scalar, optional
            Unused for closed polygons (kept for signature parity with
            *add_polyline*); there is no open start/end to pad.
        resolution : int | ArrayLike, optional
            Passed through to the surface generator (controls interpolation
            from the inner to the outer offset line).
        priority : int, default 0
            Priority of the boundary band/interface item.
        group : str, optional
            Group name for the boundary band/interface.
        add_central_cell : bool, default True
            If True, generate a finite-width band of cells traced around the
            polygon outline. If False, treat the outline as a thin interface:
            the two offset lines are pulled in to a quarter of *width* on
            each side instead of half.
        interior_group : str, optional
            If given, background points enclosed by the polygon outline are
            also added to the Voronoi diagram and tagged with this group —
            i.e. everything inside the polygon is selected as
            *interior_group*.
        interior_priority : int, optional
            Priority of the interior region. Defaults to one more than
            *priority*, so interior points take precedence over the boundary
            band.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from shapely import (
            Polygon as ShapelyPolygon,
            MultiPolygon as ShapelyMultiPolygon,
            contains,
            points as shapely_points,
        )
        from .. import extract_cells, get_cell_centers, split_lines

        if isinstance(mesh_or_points, ShapelyMultiPolygon):
            for _poly in mesh_or_points.geoms:
                self.add_polygon(
                    mesh_or_points=_poly,
                    width=width,
                    preference=preference,
                    resolution=resolution,
                    priority=priority,
                    group=group,
                    add_central_cells=add_central_cells,
                    interior_group=interior_group,
                    interior_priority=interior_priority,
                )
            return self

        if isinstance(mesh_or_points, ShapelyPolygon):
            points_in = np.asanyarray(mesh_or_points.boundary.xy)
            points_in = np.vstack([points_in, np.zeros(points_in.shape[1])]).T
        elif not isinstance(mesh_or_points, pv.PolyData):
            points_in = np.asanyarray(mesh_or_points)
        else:
            points_in = mesh_or_points.points

        # Close the ring if it isn't already closed
        if not np.allclose(points_in[0], points_in[-1]):
            points_in = np.vstack((points_in, points_in[0]))

        mesh = pv.MultipleLines(points_in)
        perc = resolution_to_perc(resolution)
        if add_central_cells:
            perc = [2.0 * perc[0] - perc[1], *perc.tolist(), 2.0 * perc[-1] - perc[-2]]
        else:
            perc = [-1.5, 0.5, 0.5, 2.5]
        # Loop over polylines
        for polyline in split_lines(mesh):
            # Remove axis from points; drop the repeated closing point and
            # keep a copy of the raw outline for the interior containment test
            points_raw = np.delete(polyline.points, self.axis, axis=1)[:-1]
            points = points_raw

            # Calculate new point coordinates if cell centers (wraps around)
            if preference == "cell":
                points = 0.5 * (points + np.roll(points, -1, axis=0))

            # Calculate forward direction vectors
            fdvec = np.diff(points, axis=0)
            fdvec = np.vstack((fdvec, fdvec[-1]))

            # Calculate backward direction vectors
            bdvec = np.diff(points[::-1], axis=0)[::-1]
            bdvec = np.vstack((bdvec[0], bdvec))

            # Calculate normal vectors
            fnorm = np.column_stack((-fdvec[:, 1], fdvec[:, 0]))
            bnorm = np.column_stack((bdvec[:, 1], -bdvec[:, 0]))
            normals = 0.5 * (fnorm + bnorm)
            normals /= np.linalg.norm(normals, axis=1)[:, None]

            # Re-insert axis
            points = np.insert(points, self.axis, 0.0, axis=1)
            normals = np.insert(normals, self.axis, 0.0, axis=1)

            if add_central_cells:
                tvec = 0.5 * width * normals
            else:
                tvec = 0.25 * width * normals

            line_a = points - tvec
            line_b = points + tvec

            # Generate structured grid with constraint cells
            plane = "yz" if self.axis == 0 else "xz" if self.axis == 1 else "xy"
            mesh_ = generate_surface_from_two_lines(line_a, line_b, plane, perc)

            # Identify constraint cells
            shape = [n - 1 for n in mesh_.dimensions if n != 1]
            constraint_ = np.ones(shape, dtype=bool)
            constraint_[: shape[0], 1:-1] = False
            constraint_ = constraint_.ravel(order="F")

            # Add to items
            if add_central_cells:
                item = MeshItem(
                    extract_cells(mesh_, ~constraint_),
                    group=group,
                    priority=priority if priority else 0,
                )
                self.items.append(item)

            item = MeshItem(extract_cells(mesh_, constraint_), group=None, priority=0)
            self.items.append(item)

            # Select background points inside the polygon and tag them as
            # their own group -- mirrors how add_circle(plain=True) seeds a
            # dense interior fill, but reuses the existing background
            # resolution instead of generating a fresh mesh.
            if interior_group is not None:
                self._polygons.append(
                    Polygon(
                        poly=ShapelyPolygon(line_a) if add_central_cells else ShapelyPolygon(points_in),
                        interior_poly=ShapelyPolygon(line_b),
                        group=group,
                        interior_group=interior_group,
                        priority=priority,
                        interior_priority=interior_priority,
                        add_central_cells=add_central_cells,
                    )
                )
        return self

    def generate_mesh(
        self,
        infinity: Optional[float] = None,
        min_length: float = 1.0e-4,
        tolerance: float = 1.0e-8,
        qhull_options: Optional[str] = None,
        orientation: Literal["CCW", "CW"] = "CCW",
    ) -> pv.UnstructuredGrid:
        """
        Generate 2D Voronoi mesh.

        Parameters
        ----------
        infinity : scalar, optional
            Value used for points at infinity.
        min_length : scalar, default 1.0e-4
            Set the minimum length of polygons' edges.
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points.
        qhull_options : str, optional
            Additional options to pass to Qhull performing the Voronoi tessellation.
            See <http://www.qhull.org/html/qh-optq.htm#qhull> for more details.
        orientation : {'CCW', 'CW'}, default 'CCW'
            Orientation of the Voronoi polygons.

        Returns
        -------
        pyvista.UnstructuredGrid
            2D Voronoi mesh.

        """
        import shapely
        from shapely import Polygon, get_coordinates

        from .. import (
            average_points,
            decimate_rdp,
            extract_boundary_polygons,
            extract_cells,
            fuse_cells,
            get_cell_centers,
        )

        groups = {}
        items = sorted(self.items, key=lambda item: abs(item.priority))

        if self.preference == "cell":
            points = get_cell_centers(self.mesh).tolist()
            group_array = self._initialize_group_array(self.mesh, groups)
            priority_array = np.full(self.mesh.n_cells, -np.inf)

        elif self.preference == "point":
            points = self.mesh.points.tolist()
            group_array = self._initialize_group_array(self.mesh.n_points, groups)
            priority_array = np.full(self.mesh.n_points, -np.inf)

        active = np.ones(len(points), dtype=bool)

        for i, item in enumerate(items):
            mesh_a = item.mesh
            points_ = get_cell_centers(mesh_a)

            # Remove out of bound points from item mesh
            mask = self.mesh.find_containing_cell(points_) != -1
            mask = cast(NDArray, mask)

            if mask.any():
                mesh_a = extract_cells(mesh_a, mask)
                points_ = points_[mask]

            # Initialize item arrays
            item_group_array = self._initialize_group_array(mesh_a, groups, item.group)
            item_priority_array = np.full(mesh_a.n_cells, abs(item.priority))

            # Disable existing points contained by item mesh and with lower (or equal) priority
            if not isinstance(mesh_a, pv.PolyData):
                idx = mesh_a.find_containing_cell(points)
                mask = np.logical_and(
                    idx != -1,
                    (
                        priority_array <= item_priority_array[idx]
                        if item.priority >= 0
                        else priority_array < item_priority_array[idx]
                    ),
                )
                active[mask] = False
                group_array[mask] = False

            # Append points to point list
            points += points_.tolist()
            active = np.concatenate((active, np.ones(len(points_), dtype=bool)))
            group_array = np.concatenate((group_array, item_group_array))
            priority_array = np.concatenate((priority_array, item_priority_array))

        points = np.delete(points, self.axis, axis=1)
        voronoi_points = points[active]
        regions, vertices = self._generate_voronoi_tesselation(
            voronoi_points, infinity, qhull_options
        )

        # Average points within minimum distance
        if min_length > 0.0:
            poly = average_points(
                pv.PolyData().from_irregular_faces(
                    np.insert(vertices, 2, 0.0, axis=-1), regions
                ),
                tolerance=min_length,
            )
            regions = poly.irregular_faces
            vertices = poly.points
            mask = np.isin(
                np.arange(len(voronoi_points)),
                poly.cell_data["vtkOriginalCellIds"],
                assume_unique=True,
                invert=True,
            )

            if mask.any():
                idx = np.arange(len(active))[active]
                active[idx[mask]] = False
                voronoi_points = voronoi_points[~mask]

        # Generate boundary polygon
        boundary_polygons = extract_boundary_polygons(
            self.mesh, fill=False, with_holes=True
        )

        if boundary_polygons is None or len(boundary_polygons) == 0:
            raise ValueError(
                "could not extract boundary polygons for the background mesh"
            )

        boundary_polygon = boundary_polygons[0]
        boundary = [
            np.delete(decimate_rdp(polygon).points, self.axis, axis=1)
            for polygon in boundary_polygon
        ]
        boundary = Polygon(boundary[0], boundary[1:])

        # Generate polygonal mesh
        points, cells = [], []
        n_points = 0

        polygons = []
        for i, region in enumerate(regions):
            polygon = Polygon(vertices[region])

            if not polygon.is_valid:
                raise ValueError(f"region {i} is not a valid polygon")

            polygon = boundary.intersection(polygon)
            polygons.append(polygon)
            points_ = get_coordinates(polygon)[:-1]
            cells += [len(points_), *(np.arange(len(points_)) + n_points)]

            # Ensure correct orientation
            signed_area = self._compute_signed_area(points_[:3])

            if (orientation == "CCW" and signed_area < 0.0) or (
                orientation == "CW" and signed_area > 0.0
            ):
                points_ = points_[::-1]

            points += list(points_)
            n_points += len(points_)

        points = self._check_point_array(points)
        mesh = pv.PolyData(points, faces=cells)
        mesh = mesh.cast_to_unstructured_grid()

        # Identify voronoi points outside the domain
        _vp = shapely.points(voronoi_points)
        _mask_vp_out = ~shapely.contains(boundary, _vp)
        _poly_vp_out = np.array(polygons)[_mask_vp_out]

        # Apply polygons selection => must account for polygons with voronoi centers
        # outside the domain but intersecting the polygon.
        active_group_array = group_array[active]
        active_priority = priority_array[active]

        for _poly in self._polygons:
            # Condition 1: Must have higher priority
            cond1 = active_priority < _poly.interior_priority
            # Condition 2.1: The polygon must contain the voronoi center
            if not _poly.add_central_cells:
                cond2 = shapely.contains_properly(_poly.poly, _vp)
            else:
                cond2 = shapely.contains_properly(_poly.interior_poly, _vp)
            if not _poly.add_central_cells:
                # Condition 2.2: voronoi center outside domain AND polygon intersects
                cond2_2 = np.zeros_like(cond1, dtype=bool)
                cond2_2[_mask_vp_out] = shapely.intersects(_poly_vp_out, _poly.poly)
                # Combine: cond1 AND (cond2_1 OR cond2_2)
                cond2 = np.logical_or(cond2, cond2_2)

            mask = np.logical_and(cond1, cond2)
            active_priority[mask] = _poly.interior_priority
            active_group_array[mask] = self._get_group_number(_poly.interior_group, groups)

        # Add the updated groups to the mesh
        mesh.cell_data["CellGroup"] = active_group_array
        mesh.user_dict["CellGroup"] = groups
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        # Add coordinates of Voronoi points
        voronoi_points = np.insert(voronoi_points, self.axis, 0.0, axis=1)
        mesh.cell_data["X"] = voronoi_points[:, 0]
        mesh.cell_data["Y"] = voronoi_points[:, 1]
        mesh.cell_data["Z"] = voronoi_points[:, 2]

        # Fuse cells, if any
        if self.fuse_cells:
            points = get_cell_centers(mesh)
            indices = [func(points) for func in self.fuse_cells]
            mesh = fuse_cells(mesh, indices)

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))

    @staticmethod
    def _compute_signed_area(points: NDArray) -> float:
        """Compute signed area of a polygon given its vertices."""
        x, y = points.T

        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    def _generate_voronoi_tesselation(
        self,
        points: ArrayLike,
        infinity: Optional[float] = None,
        qhull_options: Optional[str] = None,
    ) -> tuple[list[list[NDArray]], NDArray]:
        """
        Generate Voronoi tessalation.

        Note
        ----
        See <https://stackoverflow.com/a/43023639>.

        """
        voronoi = Voronoi(points, qhull_options=qhull_options)

        # Construct a map containing all ridges for a given point
        ridges = {}

        for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        center = voronoi.points.mean(axis=0)
        radius = infinity if infinity else np.ptp(self.mesh.points).max() * 1.0e3
        new_vertices = voronoi.vertices.tolist()
        new_regions = []

        for p1, region in enumerate(voronoi.point_region):
            vertices = voronoi.regions[region]

            if -1 not in vertices:
                new_regions.append(vertices)

            else:
                ridge = ridges[p1]
                new_region = [v for v in vertices if v >= 0]

                for p2, v1, v2 in ridge:
                    if v2 < 0:
                        v1, v2 = v2, v1

                    if v1 >= 0:
                        continue

                    t = voronoi.points[p2] - voronoi.points[p1]
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])

                    midpoint = voronoi.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = voronoi.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                # Sort region counterclockwise
                vs = np.array([new_vertices[v] for v in new_region])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_regions.append([new_region[i] for i in np.argsort(angles)])

        return new_regions, np.array(new_vertices)

    @property
    def mesh(self) -> pv.DataSet:
        """Get background mesh."""
        return self._mesh

    @property
    def axis(self) -> int:
        """Get discarded axis."""
        return self._axis

    @property
    def preference(self) -> Literal["cell", "point"]:
        """Get preference."""
        return cast(Literal["cell", "point"], self._preference)

    @property
    def fuse_cells(self) -> list[Callable]:
        """Get list of cells to fuse."""
        return self._fuse_cells

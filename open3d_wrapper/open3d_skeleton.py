from typing import Any, List, Optional, Union
import open3d as o3d
import numpy as np


class Open3DSkeleton:
    def __init__(
        self,
        points: List[o3d.geometry.TriangleMesh],
        lines: Optional[o3d.geometry.LineSet] = None,
        links: List[List[int]] = None,
    ) -> None:
        self.points = points
        self.lines = lines

        if lines is not None:
            assert links is not None, "If lines is provided, include also the links "
        self.links = links

    def update(
        self,
        points_locations: List[List[int]],
        relative: bool = False,
        new_colors: Optional[List[List[int]]] = None,
        new_links: Optional[List[List[int]]] = None,
    ) -> None:
        assert len(points_locations) == len(
            self.points
        ), "Expected the new locations to have the same number of elements of self.points!"

        if new_colors is not None:
            assert len(new_colors) == len(
                self.points
            ), "Expected the new colors to have the same number of elements of self.points!"

        for i, pt in enumerate(self.points):
            pt.translate(points_locations[i], relative=relative)
            if new_colors is not None:
                pt.paint_uniform_color(new_colors[i])

        # from docs:
        # line_set.points = o3d.utility.Vector3dVector(points)
        # line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.colors = o3d.utility.Vector3dVector(colors)

        if self.lines is not None:
            if isinstance(self.lines, LineMesh):
                # self.lines.update(points_locations, new_colors)
                raise NotImplementedError()
            else:
                self.lines.points = o3d.utility.Vector3dVector(points_locations)
                if new_links is not None:
                    self.lines.lines = o3d.utility.Vector2iVector(
                        new_links
                    )  # necessario solo se cambia self.links

                if new_colors is not None:
                    self.lines.colors = o3d.utility.Vector3dVector(new_colors)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh:
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = (
            np.array(lines)
            if lines is not None
            else self.lines_from_ordered_points(self.points)
        )
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    def update(self, positions: List[float], colors: Optional[List[float]]):
        self.points = np.asarray(positions)
        if colors is not None:
            self.colors = np.array(colors)
        self.create_line_mesh()
        return self.cylinder_segments

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length
            )
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a),
                    center=cylinder_segment.get_center(),
                )
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

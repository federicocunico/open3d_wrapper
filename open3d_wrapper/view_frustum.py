from __future__ import annotations
from typing import List
import numpy as np
import math

try:
    import open3d.cuda.pybind as o3d  # change to open3d.cpu.pybind if cuda not available
except ImportError:
    import open3d.cpu.pybind as o3d
import copy
import time


def __compute_p(pointA, pointB, pointFrom, w):
    """
    order: y,z,x
    """
    slopeX_y = (pointA[2] - pointB[2]) / (pointA[0] - pointB[0] + 0.001)

    dx = math.sqrt(w**2 / (slopeX_y**2 + 1))
    dy_x = -slopeX_y * dx

    if pointB[1] < pointA[1]:
        findX = pointFrom[2] + dx
        findY = pointFrom[0] + dy_x
    else:
        findX = pointFrom[2] - dx
        findY = pointFrom[0] - dy_x
    return [findY, pointFrom[1], findX]


def __compute_h(p, third, nose, h):
    slopep_z = (nose[1] - third[1]) / (nose[0] - third[0] + 0.001)

    dz = math.sqrt(h**2 / (slopep_z**2 + 1))
    dy = -slopep_z * dz

    y1 = p[0] + dy
    z1 = p[1] + dz

    y2 = p[0] - dy
    z2 = p[1] - dz

    return [y1, z1, p[2]], [y2, z2, p[2]]


def _compute_vf(nose, third, sizew, sizeh):
    p1 = __compute_p(nose, third, third, sizew)
    p2 = __compute_p(third, nose, third, sizew)
    p3, p4 = __compute_h(p1, third, nose, sizeh)
    p5, p6 = __compute_h(p2, third, nose, sizeh)
    return p3, p4, p5, p6


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Align a vector1 to vector2
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

        return rotation_matrix
    else:
        return np.eye(3)


class ViewFrustum:
    def __init__(self, size, vf_length, angle_w, angle_h, approximation):
        self.size = size
        self.angle_w = angle_w
        self.angle_h = angle_h
        self.vf_length = vf_length  
        self.approximation = approximation
        self.lines = o3d.geometry.LineSet()
        self.subdivisions = []

    def compute(self):
        """
        Return viewfrustum LineSet looking up
        """

        # vf_dist = 20 # self.size**2
        vf_dist = self.vf_length
        vf_dist = np.sqrt(vf_dist)
        third = [0, vf_dist, 0]  # vertical

        triangle_side_a = self.size * math.tan((self.angle_w / 2) * math.pi / 180)
        triangle_side_b = self.size * math.tan((self.angle_h / 2) * math.pi / 180)

        p1, p2, p3, p4 = _compute_vf([0, 0, 0], third, triangle_side_a, triangle_side_b)

        # points = [[0, 0, 0], third, p1, p4, p3, p2]
        points = [[0, 0, 0], third, p1, p4, p3, p2]

        # points linking
        lines = [
            # [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [2, 4],
            [2, 5],
            [3, 4],
            [3, 5],
        ]
        colors = [[1, 0, 0]] * (len(lines))
        self.lines.points = o3d.utility.Vector3dVector(points)
        self.lines.lines = o3d.utility.Vector2iVector(lines)
        self.lines.colors = o3d.utility.Vector3dVector(colors)

        ### approximation of the viewfrustum in boxes ###
        top_points = [p1, p4, p3, p2]

        # h,w of the 2d box (side view)
        box_heigth = top_points[1][2] - top_points[0][2]
        box_width = top_points[0][0] - top_points[3][0]
        offset_z = vf_dist / self.approximation
        offset_y = triangle_side_b / self.approximation
        offset_x = triangle_side_a / self.approximation
        point_corner = top_points[3]  # origin point of the box
        for sub in range(1, self.approximation + 1):
            box = o3d.geometry.TriangleMesh.create_box(box_width, offset_z, box_heigth)
            box.translate(np.asarray(point_corner) - [0, offset_z, 0])
            self.subdivisions.append(
                o3d.geometry.LineSet().create_from_triangle_mesh(box)
            )
            point_corner = [
                point_corner[0] + offset_y,
                point_corner[1] - offset_z,
                point_corner[2] + offset_x,
            ]

            box_width = 2 * triangle_side_b - 2 * sub * offset_y
            box_heigth = 2 * triangle_side_a - 2 * sub * offset_x

        self.original_vf_lines = copy.deepcopy(self.lines)
        self.original_subdivisions = copy.deepcopy(self.subdivisions)
        # self.volume = [[0,0,0], p1, p4, p3, p2, [0,0,0]]  # ORDER MATTERS

    @property
    def volume(self):
        # p1 = self.lines.points[2]
        # p2 = self.lines.points[3]
        # p3 = self.lines.points[4]
        # p4 = self.lines.points[5]
        # return [[0,0,0], p1, p4, p3, p2]
        origin = self.lines.points[0]
        third = self.lines.points[1]
        p1 = self.lines.points[2]
        p2 = self.lines.points[3]
        p3 = self.lines.points[4]
        p4 = self.lines.points[5]

        # return [origin, p1, p2, p3, p4]
        # return [origin, p1, p3, origin, origin, p4, p2, origin]
        z_off = np.asarray([0, 0, 1])
        return [p1, p3, p2, p4, p1 - z_off, p3 - z_off, p2 - z_off, p4 - z_off]

    def move_and_rotate(self, origin_point, extended_point, center_offset=None):
        """
        Rotate the static viewfrustum to align with the head-nose line
        """
        if center_offset is None:
            center_offset = [0, 0, 0]

        # calculate view line
        dist_x = origin_point[0] - extended_point[0]
        dist_z = origin_point[1] - extended_point[1]
        dist_y = origin_point[2] - extended_point[2]

        third = [
            (extended_point[0] - dist_x),
            (extended_point[1] - dist_z),
            (extended_point[2] - dist_y),
        ]

        # shift to center
        extended_point = [
            extended_point[0] + center_offset[0],
            extended_point[1] + 0,
            extended_point[2] + center_offset[1],
        ]
        third = [third[0] + center_offset[0], third[1] + 0, third[2] + center_offset[1]]

        diff = np.asarray(extended_point) - np.asarray(third)
        if (
            diff[0] != 0 or diff[2] != 0 or diff[1] > 0
        ):  # if NOT looking vertically down
            vec = np.asarray(extended_point) - np.asarray(third)

            # vec = np.asarray(third) - np.asarray(nose)
            # print(f"nose: {np.asarray(nose)}, third: {np.asarray(third)}")
            # v = np.linalg.norm(vec)
            # vstar = vec/v
            # d = vstar * 3
            # pipo = np.asarray(nose) + d
            # print(f"pipo: {np.asarray(pipo)}")
            # time.sleep(2)

            # view_frustum = skeleton_lines[id]
            # self.lines.points = copy.deepcopy(
            #     staticViewFrustum.lines.points
            # )  # reset so it'll rotate from the origin position
            self.lines.points = copy.deepcopy(self.original_vf_lines.points)
            # self.copyfrom(staticViewFrustum)

            for i, sub in enumerate(self.subdivisions):
                # self.subdivisions[i].points = staticViewFrustum.subdivisions[i].points
                self.subdivisions[i].points = self.original_subdivisions[i].points

            rad = math.atan2(vec[2], vec[0])  # plane (xy) angle of the view
            if rad != 0:
                self.lines.rotate(
                    self.lines.get_rotation_matrix_from_axis_angle((0, -rad, 0)),
                    center=(0, 0, 0),
                )  # rotate on plane
                for sub in self.subdivisions:
                    sub.rotate(
                        self.lines.get_rotation_matrix_from_axis_angle((0, -rad, 0)),
                        center=(0, 0, 0),
                    )

            self.lines.rotate(
                rotation_matrix_from_vectors(
                    [0, 1, 0], np.asarray(third) - np.asarray(extended_point)
                ),
                center=(0, 0, 0),
            )  # align with view height
            for sub in self.subdivisions:
                sub.rotate(
                    rotation_matrix_from_vectors(
                        [0, 1, 0], np.asarray(third) - np.asarray(extended_point)
                    ),
                    center=(0, 0, 0),
                )

            self.lines.translate(np.asarray(extended_point))  # move to nose position
            for sub in self.subdivisions:
                sub.translate(np.asarray(extended_point))

            # return

    @staticmethod
    def visible_points(point_cloud: np.ndarray, vfs: List[ViewFrustum]):
        ##### Selection Polygon Volume not working! #####
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # # bounding_polygon = copy.deepcopy(np.asarray(self.lines.points))
        # bounding_polygon = copy.deepcopy(np.asarray(self.volume))

        # # Create a SelectionPolygonVolume
        # vol = o3d.visualization.SelectionPolygonVolume()

        # # You need to specify what axis to orient the polygon to.
        # # I choose the "Y" axis. I made the max value the maximum Y of
        # # the polygon vertices and the min value the minimum Y of the
        # # polygon vertices.
        # vol.orthogonal_axis = "Y"
        # vol.axis_max = np.max(bounding_polygon[:, 1])
        # vol.axis_min = np.min(bounding_polygon[:, 1])

        # # Set all the Y values to 0 (they aren't needed since we specified what they
        # # should be using just vol.axis_max and vol.axis_min).
        # bounding_polygon[:, 1] = 0

        # # Convert the np.array to a Vector3dVector
        # vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

        # # Crop the point cloud using the Vector3dVector
        # cropped_pcd = vol.crop_point_cloud(pcd)

        # cropped_pts = np.asarray(cropped_pcd.points)
        # cropped_idxs = []
        # for i, pt in enumerate(point_cloud):
        #     if np.any(np.all(cropped_pts == pt, axis=1)):
        #         cropped_idxs.append(i)

        # return cropped_idxs

        #### Solution 2 based on multiple viewfrustum ####
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(point_cloud)
        for vf in vfs:
            # vf = np.asarray(vfs[nvf].lines.points)  # viewfrustum
            vf_subdivisions = np.asarray(vf.subdivisions)  # viewfrustum divisions

            pcd_c_points = []
            for subdiv in vf_subdivisions:
                subdiv_array = o3d.utility.Vector3dVector(subdiv.points)
                curr_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                    subdiv_array
                )
                pcd_c_points += pcd_c.crop(curr_bbox).points

        cropped_pts = np.asarray(pcd_c_points)
        cropped_idxs = []
        if len(cropped_pts) > 0:
            for i, pt in enumerate(point_cloud):
                if np.any(np.all(cropped_pts == pt, axis=1)):
                    cropped_idxs.append(i)

        cropped_idxs = np.unique(cropped_idxs)
        return cropped_idxs

    def copyfrom(self, vf):
        self.size = copy.deepcopy(vf.size)
        self.angle_w = copy.deepcopy(vf.angle_w)
        self.angle_h = copy.deepcopy(vf.angle_h)
        self.approximation = copy.deepcopy(vf.approximation)
        self.lines = copy.deepcopy(vf.lines)
        self.subdivisions = copy.deepcopy(vf.subdivisions)

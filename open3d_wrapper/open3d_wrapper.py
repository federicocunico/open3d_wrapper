import os
import time
from typing import Any, List, Optional, Union
import open3d as o3d
import numpy as np
from trimesh import PointCloud
from .open3d_skeleton import LineMesh, Open3DSkeleton


class Open3DWrapper:
    def __init__(self) -> None:
        self.vis: o3d.visualization.Visualizer = None
        self.window_width = None
        self.window_height = None
        self.geometries = []

    def initialize_visualizer(
        self,
        window_name: str = "Open3D",
        width: int = 960,
        height: int = 540,
        visible: bool = True,
        legacy: bool = True,
    ) -> o3d.visualization.Visualizer:
        # if legacy:
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window(
        #         window_name=window_name, width=width, height=height, visible=visible
        #     )
        # else:
        #     app = gui.Application.instance
        #     app.initialize()
        #     vis = o3d.visualization.O3DVisualizer(
        #         title=window_name, width=width, height=height
        #     )
        #     vis.show_settings = True
        #     app.add_window(vis)
        #     raise NotImplementedError()
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        self.vis = vis
        self.window_width = width
        self.window_height = height
        return vis

    def set_camera_parameters(self, camera_config_file: str = "camera_config.json") -> None:
        if self.vis is None:
            return

        if not os.path.isfile(camera_config_file):
            print("Camera parameters not found!! Press ctrl+p to generate, or invoke create_camera_parameters_config()")
            return

        wc = self.vis.get_view_control()
        # depends on your screen, when running press ctrl+p to generate a new file with visualization info
        camera_parameters = o3d.io.read_pinhole_camera_parameters(camera_config_file)
        wc.convert_from_pinhole_camera_parameters(camera_parameters)

        self.vis.poll_events()
        self.vis.update_renderer()

    def create_camera_parameters_config(self) -> None:
        tmp = Open3DWrapper()
        tmp.initialize_visualizer(window_name="View Calibration - Press CTRL+P")
        tmp.create_coordinate_system([0, 0, 0])
        tmp.wait(60)
        tmp.destroy_window()
        del tmp

    # def set_camera_transform(self, location, rotation):
    #     ctrl = self.vis.get_view_control()
    #     ctrl.translate(0,0)

    def save(self, fname):
        self.vis.capture_screen_image(fname)

    def get_current_frame(self):
        o3d_screenshot_mat = self.vis.capture_screen_float_buffer()
        # scale and convert to uint8 type
        o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        return o3d_screenshot_mat

    def create_plane(self, location_y: float = 0, size_x: float = 1, size_z: float = 1, color=None):
        offset = 0  # 0.1  # 10 cm
        h = 0.05  # 5cm
        plane = o3d.geometry.TriangleMesh.create_box(width=size_x, height=h, depth=size_z)

        # plane.compute_vertex_normals()
        if color is None:
            # color = [125 / 255, 125 / 255, 125 / 255]
            color = [242 / 255, 242 / 255, 242 / 255]
        plane.paint_uniform_color(color)

        x = -(size_x / 2)
        y = location_y - h - offset
        z = -(size_z / 2)
        plane.translate((x, y, z))  # rebase to zero + center location

        return plane

    def add_plane(self, location_y: float = 0, size_x: float = 8, size_z: float = 8, color=None):
        plane = self.create_plane(location_y, size_x, size_z, color)
        self.add_geometry(plane)

    def update(self) -> None:
        # Step 1: update geometries transforms
        for geom in self.geometries:
            self.vis.update_geometry(geom)

        # Step 2: wait for events
        self.vis.poll_events()

        # Step 3: update renderer view
        self.vis.update_renderer()

    def wait(self, seconds: int) -> None:
        # TODO: inspect self.vis.run()
        start = time.time()
        while True:
            if time.time() - start > seconds:
                break
            self.update()

    def _o3d_add_geometry(self, obj):
        # prevents resetting the view
        view_control = self.vis.get_view_control()
        pc = view_control.convert_to_pinhole_camera_parameters()

        self.vis.add_geometry(obj)

        view_control.convert_from_pinhole_camera_parameters(pc, allow_arbitrary=True)

    def add_geometry(self, geometry: Union[PointCloud, List[PointCloud]]) -> None:
        assert self.vis is not None, "Visualizer is required, try to call initialize_visualizer()"
        if isinstance(geometry, list):
            # [self.vis.add_geometry(geom) for geom in geometry]
            [self._o3d_add_geometry(geom) for geom in geometry]
            self.geometries += geometry
        else:
            self._o3d_add_geometry(geometry)
            self.geometries.append(geometry)

    # def create_box_simple(
    #     self,
    #     location: Optional[List[int]] = None,
    #     rotation: Optional[List[int]] = None,
    #     color: Optional[List[int]] = None,
    # ) -> o3d.geometry.TriangleMesh:
    #     if color is None:
    #         color = [1.0, 0.0, 0.0]  # red

    #     mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    #     mesh_box.compute_vertex_normals()
    #     mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    #     # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
    #     #     radius=0.3, height=4.0
    #     # )
    #     # mesh_cylinder.compute_vertex_normals()
    #     # mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    #     return mesh_box

    def create_box(
        self,
        size_x: float = 1.0,
        size_y: float = 1.0,
        size_z: float = 1.0,
        location: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,  # Quaternion [x, y, z, w]
        color: Optional[List[float]] = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Creates a 3D box in Open3D with specified width, height, depth, location, and rotation.

        Args:
            width (float): Width of the box.
            height (float): Height of the box.
            depth (float): Depth of the box.
            location (List[float], optional): Translation [x, y, z].
            rotation (List[float], optional): Quaternion [x, y, z, w].
            color (List[float], optional): RGB color [r, g, b].

        Returns:
            o3d.geometry.TriangleMesh: The transformed box.
        """

        if color is None:
            color = [1.0, 0.0, 0.0]  # Default red color

        # Create a unit box and scale it
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=size_x, height=size_y, depth=size_z)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color(color)

        center = np.array([0.5 * size_x, 0.5 * size_y, 0.5 * size_z])
        mesh_box.translate(-center, relative=True)  # Move to origin

        # Apply rotation (if provided), in origin
        if rotation is not None:
            if len(rotation) == 4:
                quat = np.array(rotation, dtype=np.float64)  # Ensure correct dtype
                # q_scalar_first = np.roll(quat_xyzw, shift=1)
                R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)  # Convert quaternion to rotation matrix
            else:
                R = rotation  # Assume rotation matrix
            mesh_box.rotate(R, center=(0, 0, 0))

        # Apply translation using relative=False (absolute placement)
        if location is not None:
            mesh_box.translate(location, relative=False)

        self.add_geometry(mesh_box)
        # mesh_box.update = self.update_box  # Add update method to the box object

        return mesh_box

    @staticmethod
    def update_box(box: o3d.geometry.TriangleMesh, location: np.ndarray, rotation: np.ndarray):
        """
        Updates the box's location and rotation using Open3D.

        Args:
            box (o3d.geometry.TriangleMesh): The existing box object.
            new_location (np.ndarray): Updated translation (3D) [x, y, z].
            new_rotation (np.ndarray): Updated quaternion (4D) [x, y, z, w].
        """

        # Step 1: Move box to origin before applying rotation
        center = np.array(box.get_center())  # Get current geometric center
        box.translate(-center, relative=False)  # Move to origin

        # Step 2: Apply new rotation
        if len(rotation) == 4:
            rotation = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)  # Convert quaternion to rotation matrix
        box.rotate(rotation, center=(0, 0, 0))  # Apply rotation at the origin

        # Step 3: Move back to final position
        box.translate(location, relative=False)  # Apply final translation

        return box  # Return updated box

    def create_coordinate_system(self, origin: List[int] = None, size: float = 0.6) -> o3d.geometry.TriangleMesh:
        if origin is None:
            origin = [0, 0, 0]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

        self.add_geometry(mesh_frame)
        return mesh_frame

    def create_lines(self, points, lines, colors, radius: int = 0.02, legacy: bool = True) -> None:
        if not lines:
            return
        if legacy:
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            # line_set.scale(scale)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.add_geometry(line_set)
            return line_set
        else:
            raise NotImplementedError()
            line_mesh1 = LineMesh(points, lines, colors, radius=radius)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            self.add_geometry(line_mesh1_geoms)
            return line_mesh1

    def create_sphere(
        self,
        location: Optional[List[int]] = None,
        radius: float = 1.0,
        color: Optional[List[int]] = None,
    ):

        if color is None:
            color = [1.0, 0.0, 0.0]  # red

        if location is None:
            location = [0, -1, 0]

        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.compute_vertex_normals()

        mesh_sphere.translate(location, relative=False)
        mesh_sphere.paint_uniform_color(color)

        self.add_geometry(mesh_sphere)

        return mesh_sphere

    def create_points(self, points, point_colors, radius) -> List[o3d.geometry.TriangleMesh]:
        res: List[o3d.geometry.TriangleMesh] = []
        for i, p in enumerate(points):
            if i >= len(point_colors):
                c = (np.asarray([125, 125, 125]) / 255).tolist()
            else:
                c = point_colors[i]
            s = self.create_sphere(p, radius, color=c)
            res.append(s)
        return res

    def create_skeleton(
        self,
        points: List[List[int]],
        links: Optional[List[List[int]]] = None,
        line_colors: Optional[List[List[int]]] = None,
        point_colors: Optional[List[List[int]]] = None,
        radius: float = 0.5,
        line_radius: float = 0.2,
    ) -> Open3DSkeleton:
        if point_colors is None:
            point_colors = [[125 / 255, 125 / 255, 125 / 255]] * len(points)

        if line_colors is None:
            line_colors = [[125 / 255, 125 / 255, 125 / 255]] * len(points)

        # Draw points
        mesh_pts = self.create_points(points, point_colors, radius)
        mesh_lines = self.create_lines(points, links, line_colors, radius=line_radius)

        skeleton = Open3DSkeleton(mesh_pts, mesh_lines, links)

        return skeleton

    def remove(self, target: Any):
        view_control = self.vis.get_view_control()
        pc = view_control.convert_to_pinhole_camera_parameters()

        # Notes: removes only first occurrence.
        if isinstance(target, Open3DSkeleton):
            objs = target.points
        elif isinstance(target, dict):
            objs = []
            for k, v in target.items():
                if isinstance(v, o3d.geometry.TriangleMesh):
                    if v not in objs:
                        objs.append(v)
        else:
            raise NotImplementedError()

        geom_to_remove = []
        objs_to_remove = []
        for obj in objs:
            for geom in self.geometries:
                if geom == obj or id(geom) == id(obj):
                    geom_to_remove.append(geom)
                    objs_to_remove.append(obj)
                    break
        
        if len(objs_to_remove) > 0:
            for obj in objs_to_remove:
                self.geometries.remove(obj)
        if len(geom_to_remove) > 0:
            for geom in geom_to_remove:
                self.vis.remove_geometry(geom)

        left_out = [obj for obj in objs if obj not in objs_to_remove]
        if len(left_out) > 0:
            print("Some objects were not removed: ", left_out)

        view_control.convert_from_pinhole_camera_parameters(pc, allow_arbitrary=True)

    def clear(self):
        for geom in self.geometries:
            self.vis.remove_geometry(geom)

        self.geometries.clear()

    def destroy_window(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

        self.geometries.clear()


def __test__():
    wrapper = Open3DWrapper()
    wrapper.initialize_visualizer()
    wrapper.create_coordinate_system([-2, -2, -2])

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    [wrapper.create_sphere(p, 0.1) for p in points]

    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]

    wrapper.create_lines(points, lines, colors)

    wrapper.update()
    wrapper.wait(5)
    wrapper.clear()
    wrapper.wait(5)

    wrapper.destroy_window()


def __test__lines__():
    print("Demonstrating LineMesh vs LineSet")
    # Create Line Set
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create Line Mesh 1
    points = np.array(points) + [0, 0, 2]
    line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    # Create Line Mesh 1
    points = np.array(points) + [0, 2, 0]
    line_mesh2 = LineMesh(points, radius=0.03)
    line_mesh2_geoms = line_mesh2.cylinder_segments

    o3d.visualization.draw_geometries([line_set, *line_mesh1_geoms, *line_mesh2_geoms])


def __test_skeleton__():

    FPS = 2.5
    wrapper = Open3DWrapper()
    wrapper.initialize_visualizer()
    wrapper.set_camera_parameters()

    coordinate_system = None
    skeleton = None
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]

    for i in range(10):

        # kpts position
        kpts = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        # movement across frames
        v = 1 / (i + 1)
        kpts = kpts * v

        if skeleton is None:
            skeleton = wrapper.create_skeleton(kpts, links=lines, line_colors=colors, radius=0.1)
        else:
            skeleton.update(kpts)

        # Add coordinate system
        if coordinate_system is None:
            coordinate_system = wrapper.create_coordinate_system([0, 0, 0], 1)

        wrapper.update()
        wrapper.wait(1 / FPS)

    wrapper.wait(5)
    wrapper.destroy_window()


if __name__ == "__main__":
    # __test__()
    # __test__lines__()
    __test_skeleton__()

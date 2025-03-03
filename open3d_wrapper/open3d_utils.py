import numpy as np
import open3d as o3d
from .open3d_geometry import quaternion_to_rotation_matrix


def __make(location, rotation, size_x: int = 1, size_y: int = 1, size_z: int = 1):

    if len(rotation) == 4:
        rotation = quaternion_to_rotation_matrix(rotation)

    assert np.asarray(rotation).shape == (3, 3), "Rotation must be a 3x3 matrix"

    bbox_size_x = size_x
    bbox_size_y = size_y
    bbox_size_z = size_z

    def make(x, y, z):
        return [
            [0, 0, 0],
            [x, 0, 0],
            [0, y, 0],
            [x, y, 0],
            [0, 0, z],
            [x, 0, z],
            [0, y, z],
            [x, y, z],
        ]

    # generate bbox
    coords = np.asarray(make(bbox_size_x, bbox_size_y, bbox_size_z))
    center_of_robot = np.asarray([bbox_size_x / 2, bbox_size_y / 2, bbox_size_z / 2])
    # set center of rotation to center of robot
    coords -= center_of_robot
    # apply rot
    coords = np.matmul(rotation, coords.T).T
    # reset in zero
    coords += center_of_robot

    # rebase to marker position
    # coords += np.asarray([-bbox_size_x / 2, -bbox_size_y, -bbox_size_z / 2])

    # translate
    coords += location

    coords = coords.tolist()
    return coords


def create_bbox(wrapper, location, rotation_quat, size_x: int = 1, size_y: int = 1, size_z: int = 1, line_colors=None):
    coords = __make(location, rotation_quat, size_x, size_y, size_z)

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
    if line_colors is None:
        colors = [[0, 1, 0] for i in range(len(lines))]
    elif isinstance(line_colors, list) and len(line_colors) == len(lines):
        colors = line_colors
    else:
        colors = line_colors * len(lines)

    assert np.asarray(colors).shape == (len(lines), 3), "Colors must be a list of RGB values"

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(coords)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    wrapper.add_geometry(line_set)

    return line_set


def update_bbox(line_set, location, rotation_quat, size_x: int = 1, size_y: int = 1, size_z: int = 1):
    coords = __make(location, rotation_quat, size_x, size_y, size_z)
    line_set.points = o3d.utility.Vector3dVector(coords)


def create_ray(vis, position, quaternion, length=0.1, forward_direction: str = "z", color=[1, 0, 0]):
    """
    Creates a 3D ray (LineSet) originating from the palm position and aligned with the palm's orientation.

    Parameters:
    - position: (3,) array-like -> The 3D position of the palm [x, y, z]
    - quaternion: (4,) array-like -> Rotation as [w, x, y, z] (Open3D format)
    - length: float -> Length of the ray
    - forward_direction: (3,) array-like -> Base forward direction before rotation (default is [0, 0, 1])
    - color: (3,) array-like -> Color of the ray (default is red)

    Returns:
    - line_set: Open3D LineSet representing the palm-aligned ray
    """
    if forward_direction == "z":
        forward_direction = [0, 0, 1]
    elif forward_direction == "y":
        forward_direction = [0, 1, 0]
    elif forward_direction == "x":
        forward_direction = [1, 0, 0]
    else:
        assert not isinstance(forward_direction, str), "Invalid forward_direction string"
        forward_direction = np.asarray(forward_direction)

    forward_direction = np.asarray(forward_direction).reshape(-1)

    # Convert quaternion to rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

    # Rotate the forward vector using the rotation matrix
    rotated_direction = R @ forward_direction

    # Scale the direction vector by length and translate to the palm position
    start_point = np.array(position)
    end_point = start_point + rotated_direction * length

    # Create the LineSet for visualization
    points = np.array([start_point, end_point])
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])  # Set color

    vis.add_geometry(line_set)

    return line_set


def update_ray(line_set, position, quaternion, length=0.1, forward_direction: str = "z"):
    """
    Updates an existing Open3D LineSet representing the palm-aligned ray.

    Parameters:
    - line_set: Open3D LineSet object to update
    - position: (3,) array-like -> Updated palm position [x, y, z]
    - quaternion: (4,) array-like -> Updated rotation as [w, x, y, z]
    - length: float -> Length of the ray
    - forward_direction: (3,) array-like -> Base forward direction before rotation

    Returns:
    - None (updates the existing line_set)
    """
    if forward_direction == "z":
        forward_direction = [0, 0, 1]
    elif forward_direction == "y":
        forward_direction = [0, 1, 0]
    elif forward_direction == "x":
        forward_direction = [1, 0, 0]
    else:
        assert not isinstance(forward_direction, str), "Invalid forward_direction string"
        forward_direction = np.asarray(forward_direction)

    forward_direction = np.asarray(forward_direction).reshape(-1)

    # Convert quaternion to rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)

    # Rotate the forward vector using the rotation matrix
    rotated_direction = R @ forward_direction

    # Compute new start and end points
    start_point = np.array(position)
    end_point = start_point + rotated_direction * length

    # Update LineSet points
    line_set.points = o3d.utility.Vector3dVector(np.array([start_point, end_point]))


import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def create_arrow(start, direction, length=0.2, radius=0.005, color=[1, 0, 0]):
    """
    Creates an arrow representing a vector in 3D space.
    
    Parameters:
    - start: (3,) array-like -> The starting point of the arrow [x, y, z]
    - direction: (3,) array-like -> The direction of the arrow (should be normalized)
    - length: float -> Total length of the arrow (shaft + head)
    - radius: float -> Radius of the shaft (thickness)
    - color: (3,) array-like -> Color of the arrow [R, G, B]
    
    Returns:
    - arrow: Open3D TriangleMesh object
    """
    
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize

    # Shaft and head proportions
    shaft_length = length * 0.7  # 70% of the total arrow length
    head_length = length * 0.3    # 30% for the arrowhead
    head_radius = radius * 2      # Make the arrowhead wider than the shaft

    # Create shaft (cylinder)
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=shaft_length)
    shaft.paint_uniform_color(color)

    # Move shaft so that its base is at the origin
    shaft.translate((0, 0, -shaft_length / 2), relative=True)

    # Create arrowhead (cone)
    arrowhead = o3d.geometry.TriangleMesh.create_cone(radius=head_radius, height=head_length)
    # rotate the cone so that it points along the +Z axis
    arrowhead.rotate(R.from_euler("x", 180, degrees=True).as_matrix(), center=[0, 0, 0])
    arrowhead.paint_uniform_color(color)

    # Move arrowhead to the tip of the shaft
    arrowhead.translate((0, 0, -shaft_length - head_length / 2), relative=True)

    # Combine shaft and arrowhead
    arrow = shaft + arrowhead  # Merge meshes

    # Compute rotation to align the arrow with the desired direction
    z_axis = np.array([0, 0, 1])  # Open3D cylinders/cones are aligned with +Z by default
    rotation_matrix = R.from_rotvec(np.cross(z_axis, direction) * np.arccos(np.dot(z_axis, direction))).as_matrix()
    
    # Apply rotation
    arrow.rotate(rotation_matrix, center=[0, 0, 0])

    # Move the arrow to its starting position
    arrow.translate(start, relative=False)

    return arrow


def create_cylinder_ray(vis, position, rotation, length=0.1, radius=0.005, color=[1, 0, 0]):
    """
    Creates a thick ray using a cylinder aligned with the given quaternion.

    Parameters:
    - vis: Open3D Visualizer object
    - position: (3,) array-like -> The 3D position of the ray's base [x, y, z]
    - quaternion: (4,) array-like -> Rotation as [x, y, z, w]
    - length: float -> Length of the cylinder
    - radius: float -> Radius of the cylinder (thickness)
    - forward_direction: str or (3,) array-like -> Initial forward direction ("x", "y", "z", or custom)
    - color: (3,) array-like -> Color of the cylinder [R, G, B]

    Returns:
    - cylinder: Open3D TriangleMesh representing the thick ray
    """

    if len(rotation) == 4:
        # Convert quaternion to rotation matrix using scipy (more stable)
        rotation = R.from_quat(rotation).as_matrix()

    # Create cylinder
    # cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder = create_arrow([0, 0, 0], [0, 0, 1], length=length, radius=radius, color=color)
    vis.add_geometry(cylinder)
    cylinder.paint_uniform_color(color)
    # cylinder.translate([0,0,0], relative=False)  # Move to origin for rotation
    # vis.wait(2)

    ##############
    # Rotate cylinder so it points along the X or Y axis instead of Z
    # Example: Rotate -90 degrees around Y to align it with the X-axis
    pre_rotation = R.from_euler("x", -90, degrees=True).as_matrix()  # Adjust axis as needed
    cylinder.rotate(pre_rotation, center=cylinder.get_center())
    # Move the cylinder down so that its base is at the origin
    cylinder.translate((0, -length / 2, 0), relative=True)

    # vis.wait(2)
    ##############

    # Align the cylinder with the computed direction using quaternion rotation
    # cylinder.rotate(rotation, center=cylinder.get_center())
    cylinder.rotate(rotation, center=[0, 0, 0])
    # vis.wait(2)

    # Move the **base** (not center) of the cylinder to the desired position
    rotated_base_offset = rotation @ np.array([0, -length / 2, 0])  # Base offset in rotated space
    final_position = position + rotated_base_offset  # Adjust for correct base placement

    # cylinder.translate(position, relative=False)
    cylinder.translate(final_position, relative=False)
    # vis.wait(10)

    return {"cylinder": cylinder, "last_rotation": rotation, "last_position": final_position}


def update_cylinder_ray(vis, cylinder, position, rotation, length=0.1):
    """
    Updates an existing cylinder representing the thick ray.

    Parameters:
    - vis: Open3D Visualizer object
    - cylinder: Open3D TriangleMesh object to update
    - position: (3,) array-like -> Updated position [x, y, z]
    - quaternion: (4,) array-like -> Updated rotation as [x, y, z, w]
    - length: float -> Length of the cylinder
    - forward_direction: str or (3,) array-like -> Initial forward direction

    Returns:
    - None (updates the existing cylinder)
    """

    if len(rotation) == 4:
        # Convert quaternion to rotation matrix using scipy (more stable)
        rotation = R.from_quat(rotation).as_matrix()

    #vis.wait(10)
    cylinder_mesh, old_rot = cylinder["cylinder"], cylinder["last_rotation"] # , cylinder["last_position"]

    # revert to original position
    cylinder_mesh.translate([0,0,0], relative=True)
    cylinder_mesh.translate((0, -length / 2, 0), relative=True)
    # vis.wait(5)
    R_inverse = np.linalg.inv(old_rot)
    cylinder_mesh.rotate(R_inverse, center=[0, 0, 0])
    # vis.wait(5)
    #cylinder_mesh.translate((0, -length / 2, 0), relative=True)
    # vis.wait(5)

    # Align the cylinder with the computed direction using quaternion rotation
    # cylinder.rotate(rotation, center=cylinder.get_center())
    cylinder_mesh.rotate(rotation, center=[0, 0, 0])
    # vis.wait(5)

    # Move the **base** (not center) of the cylinder to the desired position
    rotated_base_offset = rotation @ np.array([0, -length / 2, 0])  # Base offset in rotated space
    final_position = position + rotated_base_offset  # Adjust for correct base placement

    # cylinder.translate(position, relative=False)
    cylinder_mesh.translate(final_position, relative=False)
    # vis.wait(10)

    # # Apply rotation and translation updates
    # cylinder_mesh.translate([0, 0, 0], relative=False)  # Move to origin for rotation
    # cylinder_mesh.translate((0, -length / 2, 0), relative=True)
    # # vis.wait(5)

    # # reset rotation to identity
    # R_inverse = np.linalg.inv(old_rot)
    # # cylinder_mesh.rotate(R_inverse, center=cylinder_mesh.get_center())
    # cylinder_mesh.rotate(R_inverse, center=[0, 0, 0])
    # # vis.wait(5)

    # # rotate to new rotation
    # # cylinder_mesh.rotate(rotation, center=cylinder_mesh.get_center())
    # cylinder_mesh.rotate(rotation, center=[0, 0, 0])
    # # vis.wait(5)
    # rotated_position = np.dot(rotation, position)
    # cylinder_mesh.translate(rotated_position, relative=False)
    # # vis.wait(5)

    return {"cylinder": cylinder_mesh, "last_rotation": rotation}

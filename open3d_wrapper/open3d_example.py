import open3d.cpu.pybind as o3d
import numpy as np
from tqdm import tqdm
import time
import os
import matplotlib.cm as cm
import matplotlib.colors as mtlcol
from scipy.spatial.transform import Rotation as R
import argparse
import itertools
import json


parser = argparse.ArgumentParser(description="Open3D-based visualization.")
# parser.add_argument('--config', type=str, default="configs/basic_config.json", help='path to the json configuration file')
parser.add_argument("robot_state", type=str, help="json file with spot state")
parser.add_argument("optitrack", type=str, help="csv file with optitrack data")
parser.add_argument("--not_visible", action="store_false", help="if False, hid view open3d window")
parser.add_argument("--add-floor", action="store_true", help="if True, add plane to 3d model")
args = parser.parse_args()


if __name__ == "__main__":

    scene_size = [5, 5]  # metri

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D - Viewfrustum", visible=args.not_visible)

    floor = o3d.geometry.TriangleMesh.create_box(scene_size[0], 0.0001, scene_size[1])
    floor.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(floor)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(1, [0, 0, 0])  # x=red, y=green, z=blue
    vis.add_geometry(mesh)

    # render_option = vis.get_render_option()
    # render_option.light_on = False

    with open(args.robot_state, "r") as f:
        data = json.load(f)

    session = read_optitrack_csv(args.optitrack)

    fps_array = []
    print("Reading log files ...")
    vis.poll_events()
    vis.update_renderer()
    # vis.run()

    robot_points = []

    for i in range(12):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate([0.0, 0.0, 0.0], relative=False)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 1, 0.1])
        vis.add_geometry(sphere, reset_bounding_box=False)
        robot_points.append(sphere)

    for entry_idx, frame in enumerate(data):
        start_time = time.perf_counter()

        skeleton = frame["Joints"]
        skeleton = np.array(skeleton)
        skeleton = np.reshape(skeleton, (12, 3))

        # spot_front = frame['SpotFront'][1:]
        # spot_back = frame['SpotBack'][1:]

        print(entry_idx)
        for i, point in enumerate(skeleton):
            robot_points[i].translate(point, relative=False)
            vis.update_geometry(robot_points[i])

        # time.sleep(0.1)
        vis.poll_events()
        vis.update_renderer()

        end_time = time.perf_counter()
        fps_array.append(end_time - start_time)

        if False:
            vis.capture_screen_image(os.path.join(args.out_frames_root, f"{args.exp}_frame_{entry_idx:04d}.jpg"))

    vis.destroy_window()
    print(f"mean FPS: {1/np.mean(fps_array)}")

import os
from typing import Dict
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from .open3d_skeleton import Open3DSkeleton
from .open3d_wrapper import Open3DWrapper


def visualize_session(
        session: OptitrackSession, 
        detect_collisions: bool = False, 
        save_frame: bool = False, 
        save_video: bool = False, 
        save_video_fname: str = "video.mp4",
        visible: bool = True,
        force: bool = False):
    if not force:
        if os.path.isfile(save_video_fname):
            print(f"Already done {save_video_fname}")
            return

    MAX_FPS = 120
    skeleton_radius = 0.01
    show_spot_mesh = False
    collision_th = 0.05

    out_frames_folder = "visualization_images"
    os.makedirs(out_frames_folder, exist_ok=True)

    curr_skeletons: Dict[str, OptitrackSkeleton] = session.skeletons
    curr_rigidbodies: Dict[str, OptitrackRigidBody] = session.rigid_bodies
    n_frames = session.max_frames

    print(f"Found {len(curr_skeletons)} skeletons")
    print(f"Found {len(curr_rigidbodies)} rigid bodies")
    print(f"Total number of frames: {n_frames}")

    wrapper = Open3DWrapper()
    wrapper.initialize_visualizer(window_name="OptiTrack", width=1920, height=1080, visible=visible)
    wrapper.set_camera_parameters()
    wrapper.add_plane()

    ##### Model, optional
    if show_spot_mesh:
        # spot_mesh = o3d.io.read_triangle_mesh("models/spot_model_simple__working.obj", True)
        spot_mesh = o3d.io.read_triangle_mesh("models/spot.obj", True)
        spot_mesh.compute_triangle_normals()
        wrapper.add_geometry(spot_mesh)
        last_rot = np.eye(3)
    #####
    
    #### Save Video, optional
    out = None
    ####

    spot_bbox = None
    coordinate_system = None

    objs: Dict[str, Open3DSkeleton] = {}

    frame_idx = 0
    # for i in range(n_frames):
    for i in tqdm(range(n_frames), "Rendering"):
        if i % 4 != 0:
            continue
        skeletons_dict, rigidbodies_dict = session.collect(i)
        # for the moment I just delete one of the two spot points
        # del rigidbodies_dict['spot_back']

        for sk_name, sk in skeletons_dict.items():
            sk_positions = sk[0]
            sk_rots = sk[1]

            sk_positions = [p.tolist() for p in sk_positions]
            if sk_name not in objs:
                objs[sk_name] = wrapper.create_skeleton(
                    points=sk_positions,
                    links=curr_skeletons[sk_name].links,
                    point_colors=curr_skeletons[sk_name].colors,
                    radius=skeleton_radius,
                )
            else:
                objs[sk_name].update(sk_positions)

        for rb_name, rb in rigidbodies_dict.items():
            rb_pos = rb[0]
            rb_rot = rb[1]
            spot_points = rb[2]
            if spot_points is None:
                print("No spot points found")
                continue

            ##### Model, optional
            if show_spot_mesh:
                if rb_rot is None:
                    # rb_rot = np.eye(3)
                    rb_rot = last_rot
                rb_rot_np = np.asarray(rb_rot)
                # _pos = rb_pos - np.asarray([0, 0.700/2, 0])
                _pos = rb_pos - np.asarray([0, 0.550/2, 0])
                spot_mesh.translate(_pos.tolist(), relative=False)
                spot_mesh.rotate(np.linalg.inv(last_rot))
                spot_mesh.rotate(rb_rot_np)
                last_rot = rb_rot_np
            #####

            spot_links = curr_rigidbodies[rb_name].spot_links
            spot_colors = curr_rigidbodies[rb_name].spot_colors

            if rb_name not in objs:
                objs[rb_name] = wrapper.create_skeleton(
                    # points=rb_pos, point_colors=[[1, 1, 0]], radius=0.1
                    points=spot_points,
                    links=spot_links,
                    point_colors=spot_colors,
                    radius=skeleton_radius,
                )
                # spot_bbox = create_bbox(wrapper, rb_pos, rb_rot)
            else:
                # objs[rb_name].update(rb_pos)
                objs[rb_name].update(spot_points)
                # update_bbox(spot_bbox, rb_pos, rb_rot)

        if detect_collisions:
            collision_occurred, distances = detect_collision_from_visualizer(
                wrapper, skeletons_dict, rigidbodies_dict, threshold=collision_th
            )
            if collision_occurred:
                # print("Collision at frame ", i, ". Distances: ", distances)
                print("Collision at frame ", i)

        # Add coordinate system
        if coordinate_system is None:
            coordinate_system = wrapper.create_coordinate_system([0, 0, 0], 0.5)
        wrapper.update()
        if save_frame:
            wrapper.save(f"{out_frames_folder}/{str(frame_idx).zfill(6)}.jpg")
        if save_video:
            frame = wrapper.get_current_frame()
            if out is None:
                frameSize = (frame.shape[1], frame.shape[0])
                out = cv2.VideoWriter(save_video_fname, cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1
        wrapper.wait(1 / MAX_FPS)

    if save_video:
        out.release()
    wrapper.destroy_window()

# def raw_visualize_session(session: OptitrackSession):
#     FPS = 120
#     SPEED = 1.0
#     assert SPEED >= 1.0, "Need speed > 1.0"
#     frames = session.max_frames

#     wrapper = Open3DWrapper()
#     wrapper.initialize_visualizer(window_name="OptiTrack")
#     wrapper.set_camera_parameters()
#     coordinate_system = None

#     objects: Dict[str, Open3DSkeleton] = {}

#     step = max(1, math.floor(((SPEED - 1) * 100)))
#     for i in tqdm(range(0, frames, step)):
#         curr_entities = session.collect_entities(i)
#         for e in curr_entities:
#             e_name = e.name

#             position = e.get_position(i)
#             if position is None:
#                 if e_name in objects:
#                     obj = objects[e_name]
#                     wrapper.remove(obj)
#                     objects.pop(e_name)
#                 continue

#             if e_name not in objects:
#                 new_object: Open3DSkeleton = wrapper.create_skeleton(
#                     position, radius=0.05
#                 )  # skeleton of one position, allows for update() function
#                 objects[e_name] = new_object
#             else:
#                 # print(f"Updating i={i}")
#                 curr = objects[e_name]
#                 curr.update(position)

#         # Add coordinate system
#         if coordinate_system is None:
#             coordinate_system = wrapper.create_coordinate_system([0, 0, 0], 0.5)

#         wrapper.update()
#         wrapper.wait(1 / FPS)

#         # wrapper.clear()
#         # objects.clear()

#     print("Done")
#     wrapper.wait(5)
#     wrapper.destroy_window()


def __test__():
    # session = read_optitrack_csv(args.optitrack_data)
    # # session = load_optitrack_data(csv_file.replace(".csv", ".json"))
    # # raw_visualize_session(session)
    # # visualize_session(session, save_frame=True)

    # # with open("data\\rotated_capture.json", 'r') as f:
    # # with open("data\\processed_capture_2.json", 'r') as f:
    # with open(args.spot_state, 'r') as f:
    #     spot_data = json.load(f)
    # # reformatting data
    # spot_data = {int(spot_data[i]['FrameNumber']):spot_data[i] for i in range(len(spot_data))}

    # visualize_session(session, spot_data, save_frame=args.video)

    # compatibility
    from main_visualization import main

    main()


if __name__ == "__main__":
    __test__()

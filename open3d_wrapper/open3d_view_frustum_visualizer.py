from typing import Dict, List, Optional
import open3d as o3d
import cv2
import numpy as np
from tqdm import tqdm
from .open3d_skeleton import Open3DSkeleton
from .open3d_wrapper import Open3DWrapper
from .view_frustum import ViewFrustum
from scipy.spatial.transform import Rotation


def project_3d_to_2d(objectPoints, cam_loc, cam_rot):
    if isinstance(cam_rot, Rotation):
        cam_rot = cam_rot.as_matrix()
    if cam_rot.size == 9:
        cam_rot = cv2.Rodrigues(cam_rot)[0]
    cam_rot = np.asarray(cam_rot).astype(np.float32)
    if cam_rot.size == 3:
        cam_rot = cam_rot.reshape(1, 3)
    cam_loc = np.asarray(cam_loc).reshape(1, 3).astype(np.float32)
    cameraMatrix = np.asarray(
        [
            [256.1049499511719, 0, 316.7681884765625],
            [0, 255.5893096923828, 232.4761505126953],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    dist_coeff = np.asarray([0, 0, 0, 0, 0], dtype=np.float32)
    projection = cv2.projectPoints(
        objectPoints, cam_rot, cam_loc, cameraMatrix, dist_coeff
    )
    pts = projection[0].reshape(-1, 2)
    return pts


def visible_points_by_camera(
    skeleton_points: np.ndarray,
    cam_loc: np.ndarray,
    cam_rot: np.ndarray,
    image_w: int = 640,
    image_h: int = 480,
):
    projected_points = project_3d_to_2d(skeleton_points, cam_loc, cam_rot)
    pts_larger_w = np.where(projected_points[:, 0] > image_w)
    pts_smaller_w = np.where(projected_points[:, 0] < 0)
    pts_larger_h = np.where(projected_points[:, 1] > image_h)
    pts_smaller_h = np.where(projected_points[:, 1] < 0)

    invalid_pts = np.unique(
        np.concatenate(
            (pts_larger_w, pts_smaller_w, pts_larger_h, pts_smaller_h), axis=1
        )
    )

    mask = np.ones(skeleton_points.shape[0], np.bool)
    mask[invalid_pts] = 0
    visible_pts_idxs = np.where(mask > 0)[0]

    # visible_pts = skeleton_points[visible_pts_idxs]
    # if len(visible_pts) == 0:
    #     return []
    if len(visible_pts_idxs) == 0:
        return []
    return visible_pts_idxs


def view_frustum_visualize(
    session: OptitrackSession,
    cam_str: str = "hand",
    save_video: bool = False,
    save_video_fname: str = "vf_culling.mp4",
):
    skeleton_radius = 0.01
    curr_skeletons: Dict[str, OptitrackSkeleton] = session.skeletons
    curr_rigidbodies: Dict[str, OptitrackRigidBody] = session.rigid_bodies
    n_frames = session.max_frames

    print(f"Found {len(curr_skeletons)} skeletons")
    print(f"Found {len(curr_rigidbodies)} rigid bodies")
    print(f"Total number of frames: {n_frames}")

    wrapper = Open3DWrapper()
    wrapper.initialize_visualizer(
        window_name="OptiTrack", width=1920, height=1080, visible=not save_video
    )
    wrapper.set_camera_parameters()
    wrapper.add_plane()
    wrapper.create_coordinate_system([0, 0, 0], 0.5)

    objs: Dict[str, Open3DSkeleton] = {}

    frame_idx = 0
    out_video = None
    sk_colors = None
    cams_objs = None
    skip = False

    vfs = None

    for i in tqdm(range(n_frames)):
        if i % 4 != 0:
            continue
        skeletons_dict, rigidbodies_dict = session.collect(i)

        for rb_name, rb in rigidbodies_dict.items():
            # rb_pos = rb[0]
            rb_rot = rb[1]
            spot_points = rb[2]
            if rb_rot is None:
                print("Cannot use rigid body rotation")
                skip = True
                break

            ############################

            # cam_pts, view_frustums = spot_cameras_compute(spot_points)
            objectPoints = np.asarray(skeletons_dict["atoaiari_baseline"][0])
            # # cam_rot = [0, 0, 0]
            # # cam_rot = spot_side_rotation(objectPoints)
            # cam_rot = cv2.Rodrigues(np.asarray(rb_rot))[0]
            # cam_loc = cam_pts["cam_hand"]
            # print(cam_rot)

            # indexes = visible_points(objectPoints, cam_loc, cam_rot)
            # # if len(indexes) == 0:
            # #     print("No visible points")
            # sk_colors = [
            #     [0, 1, 0] if i in indexes else [1, 0, 0]  # Green if visible, red otherwise
            #     for (i, _) in enumerate(objectPoints)
            # ]

            spot_links = curr_rigidbodies[rb_name].spot_links
            spot_colors = curr_rigidbodies[rb_name].spot_colors

            if vfs is None:
                size_meters = [2.0, 2.0, 2.0]
                # approximation = [15, 10, 5]
                approximation = [15, 15, 15]
                vf_length = 15 # meters

                fov_x = None  # 60
                fov_y = None  # 60
                vfs: List[ViewFrustum] = create_view_frustums(
                    size_meters=size_meters,
                    approximation=approximation,
                    vf_length=vf_length,
                    fov_x=fov_x,
                    fov_y=fov_y,
                )
                wrapper.add_geometry(vfs[-1].lines)
            cam_location, cam_location_ext, cam_rotation = spot_cam_info(
                spot_points, cam_str
            )

            [
                vf.move_and_rotate(
                    origin_point=cam_location,
                    extended_point=cam_location_ext,
                    center_offset=[0, 0, 0],
                )
                for vf in vfs
            ]

            indexes = ViewFrustum.visible_points(objectPoints, vfs)

            sk_colors = [
                [0, 1, 0]
                if i in indexes
                else [1, 0, 0]  # Green if visible, red otherwise
                for (i, _) in enumerate(objectPoints)
            ]

            # show camera location
            if frame_idx == 0 or cams_objs is None:
                # wrapper.add_geometry(wrapper.create_sphere(cam_loc, 0.01, [1, 165/255, 0]))  # add camera as orgnge sphere
                cams_objs = [
                    wrapper.create_sphere(cam_loc, 0.01, [1, 165 / 255, 0])
                    for cam_loc in [cam_location, cam_location_ext]
                ]  # add camera as orgnge sphere
                cam_line_set = wrapper.create_lines(
                    [cam_location, cam_location_ext], [[0, 1]], [[1, 165 / 255, 0]]
                )
            else:
                [
                    cams_objs[i].translate(cam_loc, relative=False)
                    for (i, cam_loc) in enumerate([cam_location, cam_location_ext])
                ]
                cam_line_set.points = o3d.utility.Vector3dVector(
                    [cam_location, cam_location_ext]
                )

            ############################
            if rb_name not in objs:
                objs[rb_name] = wrapper.create_skeleton(
                    points=spot_points,
                    links=spot_links,
                    point_colors=spot_colors,
                    radius=skeleton_radius,
                )
            else:
                objs[rb_name].update(spot_points)

        if skip:
            skip = False
            continue

        # Show human skeletons
        for sk_name, sk in skeletons_dict.items():
            sk_positions = [p.tolist() for p in sk[0]]
            if sk_name not in objs:
                if sk_colors is None:
                    sk_colors = [[0, 1, 0] for _ in sk_positions]
                objs[sk_name] = wrapper.create_skeleton(
                    points=sk_positions,
                    links=curr_skeletons[sk_name].links,
                    point_colors=sk_colors,
                    radius=skeleton_radius,
                )
            else:
                objs[sk_name].update(sk_positions, new_colors=sk_colors)

        # # show camera location
        # if frame_idx == 0 or cams_objs is None:
        #     # wrapper.add_geometry(wrapper.create_sphere(cam_loc, 0.01, [1, 165/255, 0]))  # add camera as orgnge sphere
        #     cams_objs = [wrapper.create_sphere(cam_loc, 0.01, [1, 165/255, 0]) for (k, cam_loc) in cam_pts.items()] # add camera as orgnge sphere
        # else:
        #     [cams_objs[i].translate(cam_loc, relative=False) for (i,(k, cam_loc)) in enumerate(cam_pts.items())]

        wrapper.update()

        frame_idx += 1
        # wrapper.wait(1 / 120)
        wrapper.wait(1 / 45)
        # wrapper.wait(100)
        # wrapper.wait(20)

        if save_video:
            frame = wrapper.get_current_frame()
            if out_video is None:
                frameSize = (frame.shape[1], frame.shape[0])
                out_video = cv2.VideoWriter(
                    save_video_fname, cv2.VideoWriter_fourcc(*"mp4v"), 30, frameSize
                )
            out_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if save_video:
        out_video.release()
    wrapper.destroy_window()

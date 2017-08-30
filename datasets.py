import lib.pySceneNetRGBD.scenenet_pb2 as sn
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import math

def normalize(v):
    return v/np.linalg.norm(v)

def world_to_camera_with_pose(view_pose):
    lookat_pose = position_to_np_array(view_pose.lookat)
    camera_pose = position_to_np_array(view_pose.camera)
    up = np.array([0,1,0])
    R = np.diag(np.ones(4))
    R[2,:3] = normalize(lookat_pose - camera_pose)
    R[0,:3] = normalize(np.cross(R[2,:3],up))
    R[1,:3] = -normalize(np.cross(R[0,:3],R[2,:3]))
    T = np.diag(np.ones(4))
    T[:3,3] = -camera_pose
    return R.dot(T)

def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))

def camera_intrinsic_transform(vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def position_to_np_array(position,homogenous=False):
    if not homogenous:
        return np.array([position.x,position.y,position.z])
    return np.array([position.x,position.y,position.z,1.0])

def interpolate_poses(start_pose,end_pose,alpha):
    assert alpha >= 0.0
    assert alpha <= 1.0
    camera_pose = alpha * position_to_np_array(end_pose.camera)
    camera_pose += (1.0 - alpha) * position_to_np_array(start_pose.camera)
    lookat_pose = alpha * position_to_np_array(end_pose.lookat)
    lookat_pose += (1.0 - alpha) * position_to_np_array(start_pose.lookat)
    timestamp = alpha * end_pose.timestamp + (1.0 - alpha) * start_pose.timestamp
    pose = sn.Pose()
    pose.camera.x = camera_pose[0]
    pose.camera.y = camera_pose[1]
    pose.camera.z = camera_pose[2]
    pose.lookat.x = lookat_pose[0]
    pose.lookat.y = lookat_pose[1]
    pose.lookat.z = lookat_pose[2]
    pose.timestamp = timestamp
    return pose

def pose_to_euler(pose):
    camera = position_to_np_array(pose.camera)
    lookat = position_to_np_array(pose.lookat)

    return camera - lookat

def trajectory_to_paths(root, traj, view):
    paths = {}
    for type in ['photo', 'depth', 'instance']:
        filetype = 'jpg' if type == 'photo' else '.png'
        paths[type] = os.path.join(root, traj.render_path, type, '{}.{}'.format(view.frame_num, filetype))
    return paths

class SceneNetRGBD(data.Dataset):

    def __init__(self, data_root, protobuf_file, sequence_length=10, rgb=True, camera_state=True): #  depth=True, instance_map=True, segmentation_map=True, flow=True,
        loader = sn.Trajectories()
        try:
            with open(protobuf_file, 'rb') as f:
                loader.ParseFromString(f.read())
        except IOError:
            print('Scenenet protobuf data not found at location')
            print('Please ensure you have copied the pb file to the data directory')

        self.sequence_length = sequence_length
        self.data_root = data_root
        self.trajectories = loader.trajectories
        self.camera_intrinsics = camera_intrinsic_transform()
        self.instances = []
        self.instance_shapes = [(sequence_length, 3, 240, 320), (sequence_length, 10)]

        for t_id, traj in enumerate(self.trajectories):
            for v_id, view in enumerate(traj.views[:-sequence_length]):
                self.instances.append((t_id, v_id))

    def __getitem__(self, index):

        trajectory_id, start_view = self.instances[index]
        trajectory = self.trajectories[trajectory_id]
        poses = []
        frames = np.zeros(self.instance_shapes[0], dtype=np.uint8)

        # compute input image frames
        for i, view in enumerate(trajectory.views[start_view:start_view + self.sequence_length]):
            paths = trajectory_to_paths(self.data_root, trajectory, view)
            image = np.asarray(Image.open(paths['photo']))
            frames[i] = image.transpose((2, 0, 1))

            ground_truth_pose = interpolate_poses(view.shutter_open, view.shutter_close, 0.5)
            poses.append(pose_to_euler(ground_truth_pose))

        # compute target poses
        targets = np.zeros(self.instance_shapes[1])
        for i, (from_pose, to_pose) in enumerate(zip(poses[:-1], poses[1:]), 1):
            targets[i][:3] = from_pose - to_pose

        return frames, targets

    def __len__(self):
        return len(self.instances)


if __name__ == '__main__':
    ds = SceneNetRGBD('/mnt/not_backed_up/scenenetrgbd/val', '/mnt/not_backed_up/scenenetrgbd/scenenet_rgbd_val.pb')
    print(len(ds))
    frames, targets = ds[400]

    print(frames.shape, targets.shape)
    print(targets)

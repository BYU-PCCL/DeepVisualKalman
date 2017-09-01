import lib.pySceneNetRGBD.scenenet_pb2 as sn
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image
import os
import os.path
import math
from threading import Lock
from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
from collections import defaultdict
import pyquaternion as pyq
import matplotlib.pyplot as plt




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

def logq(q):
    n = np.linalg.norm(q.vector)
    return 2 * np.arctan2(n, q.scalar) * q.vector / n


class SceneNetRGBD(data.Dataset):

    load_lock = Lock()
    trajectories = {}
    instances = defaultdict(list)

    def __init__(self, data_root, protobuf_glob, sequence_length, crop_size=128, rgb=True, camera_state=True): #  depth=True, instance_map=True, segmentation_map=True, flow=True,
        self.finite_difference_degree = 4

        # Load the trajectories only once for the entire duration of the program
        SceneNetRGBD.load_lock.acquire()
        if protobuf_glob not in SceneNetRGBD.trajectories:
            try:
                loader = sn.Trajectories()
                protobuf_files = glob(protobuf_glob)

                for protobuf_file in tqdm(protobuf_files, leave=False, desc='Loading Protobufs'):
                    with open(protobuf_file, 'rb') as f:
                        loader.ParseFromString(f.read())

                    if protobuf_glob not in SceneNetRGBD.trajectories:
                        SceneNetRGBD.trajectories[protobuf_glob] = loader.trajectories
                    else:
                        SceneNetRGBD.trajectories[protobuf_glob].extend(loader.trajectories)

                for t_id, traj in enumerate(SceneNetRGBD.trajectories[protobuf_glob]):
                    for v_id, view in enumerate(traj.views[:-sequence_length + 1 + self.finite_difference_degree]):
                        SceneNetRGBD.instances[protobuf_glob].append((t_id, v_id))

            except IOError:
                print('Scenenet protobuf data not found at location')
                print('Please ensure you have copied the pb file to the data directory')

        SceneNetRGBD.load_lock.release()

        self.sequence_length = sequence_length
        self.data_root = data_root
        self.camera_intrinsics = camera_intrinsic_transform()
        self.crop_size = crop_size
        self.protobuf_glob = protobuf_glob

        state_dimension = 10  # inertial_pos_x, inertial_pos_y, inertial_pos_z,
                              # body_fixed_vel_x, body_fixed_y, body_fixed_vel_z,
                              # inertial_to_body_quaterion_w,
                              # inertial_to_body_quaterion_x,
                              # inertial_to_body_quaterion_y,
                              # inertial_to_body_quaterion_z

        self.instance_shapes = [(3, sequence_length, crop_size, crop_size), (sequence_length, state_dimension)]

        assert len(self) > 0, 'There are no elements in this dataset'


    def __getitem__(self, index):
        trajectory_id, start_view = SceneNetRGBD.instances[self.protobuf_glob][index]
        trajectory = SceneNetRGBD.trajectories[self.protobuf_glob][trajectory_id]
        xs = []
        qs = []
        frames = np.zeros(self.instance_shapes[0], dtype=np.uint8)
        targets = np.zeros(self.instance_shapes[1], dtype=np.float32)

        # compute input image frames
        for i, view in enumerate(trajectory.views[start_view:start_view + self.sequence_length + self.finite_difference_degree]):
            # only load frames for self.sequence length, but min_sequence_length worth of poses
            if i < self.sequence_length:
                paths = trajectory_to_paths(self.data_root, trajectory, view)
                img = Image.open(paths['photo'])
                img = img.crop([0, 0, self.crop_size, self.crop_size])
                img = np.asarray(img)
                frames[:, i] = img.transpose((2, 0, 1))

            ground_truth_pose = interpolate_poses(view.shutter_open, view.shutter_close, 0.5)

            T = world_to_camera_with_pose(ground_truth_pose)
            position = T[:3, 3]
            inertial_to_body_quaterion = pyq.Quaternion(matrix=T)
            xs.append(position)
            qs.append(inertial_to_body_quaterion)

        xd = []
        xdd = []
        qd = []
        dt = 1.

        # compute target poses for the entires with photos
        for i in range(self.sequence_length):
            if i <= 3:
                # Forward differencing (wikipedia finite difference coefficients)
                xd.append((-11./6. * xs[i] + 3 * xs[i+1] - 1.5 * xs[i + 2] + 1./3. * xs[i + 3])/dt)
                xdd.append((2. * xs[i] - 5 * xs[i+1] + 4 * xs[i + 2] - 1 * xs[i + 3])/(dt*dt))

                omega = logq(qs[i + 1] * qs[i].inverse) / dt
                qd.append(omega)

            else:
                # Forward differencing (wikipedia finite difference coefficients)
                xd.append((11/.6 * xs[i] - 3 * xs[i-1] + 1.5 * xs[i - 2] - 1./3. * xs[i - 3])/dt)
                xdd.append((2. * xs[i] - 5 * xs[i - 1] + 4 * xs[i - 2] - 1 * xs[i - 3]) / (dt * dt))

                omega = logq(qs[i] * qs[i - 1].inverse) / dt
                qd.append(omega)

            targets[i][0:3] = xs[i]
            targets[i][3:6] = qs[i].rotate(xd[i])
            targets[i][6:10] = qs[i].elements

            # put xdd in the body fixed coordinate frame
            xdd[i] = qs[i].rotate(xdd[i])

        # TODO: are you handling error properly yet?

        return frames, np.array(xd), np.array(xdd), np.array(qd), targets[-1].copy(), targets

    def __len__(self):
        return len(SceneNetRGBD.instances[self.protobuf_glob])


if __name__ == '__main__':
    sequence_length = 100
    ds = SceneNetRGBD('/mnt/pccfs/not_backed_up/scenenetrgbd/val',
                      '/mnt/pccfs/not_backed_up/scenenetrgbd/scenenet_rgbd_val.pb',
                      sequence_length)

    print(len(ds))
    (frames, xd, xdd, omegas), targets = ds[400]

    print(frames.shape, targets.shape)
    print(targets)

    plt.figure(1)
    plt.subplot(411)
    plt.plot(range(sequence_length), omegas)
    plt.legend(['x', 'y', 'z'])
    plt.title('omega')

    plt.subplot(412)
    plt.plot(range(sequence_length), xd)
    plt.legend(['x', 'y', 'z'])
    plt.title('vel')

    plt.subplot(413)
    plt.plot(range(sequence_length), xdd)
    plt.legend(['x', 'y', 'z'])
    plt.title('accel')

    print(targets.shape)

    plt.subplot(414)
    plt.plot(range(sequence_length), targets[:, 6:])
    plt.legend(['w', 'x', 'y', 'z'])
    plt.title('quaternion')
    plt.show()

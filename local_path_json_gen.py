import os
import json
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from datetime import datetime
from multiprocessing import Pool, cpu_count


# 1. 读取 pose.txt
def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            timestamp = str(parts[0])
            x, y, z = map(float, parts[1:4])
            roll, pitch, yaw = map(float, parts[4:7])
            poses.append((timestamp, np.array([x, y, z]), np.array([roll, pitch, yaw])))
    return poses

# 2. 构造4x4变换矩阵
def pose_to_matrix(xyz, rpy):
    rotation = R.from_euler('xyz', rpy, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = xyz
    return T

# 3. 读取点云（.bin）
def load_pointcloud(bin_path):
    pointcloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    return pointcloud[:, :3]  # 只用x,y,z

def read_pcd_with_intensity(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 提取PCD文件头信息并查找"DATA ascii"部分
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("DATA ascii"):
            header_end = i + 1  # "DATA ascii"下一行是数据部分
            break

    # 读取点云数据部分
    data_lines = lines[header_end:]
    
    points = []
    index = 0
    for line in data_lines:
        # 跳过注释行
        index +=1
        if index<12:
            continue
        # 将每行数据分割并转换为浮点数
        point_data = list(map(float, line.split()))
        points.append(point_data)

    # 将点云数据转换为 NumPy 数组
    points = np.array(points)

    # 返回点云数据 (x, y, z, intensity, ring)
    return points[:,:3]

# 4. 主处理逻辑
def accumulate_pointclouds(pose_file, lidar_folder, num_fusion):
    poses = load_poses(pose_file)
    num_frames = len(poses)
    print('process lidar_folder:',lidar_folder)

    for ref_idx in range(0, num_frames):
        
        ref_pose = poses[ref_idx]
        ref_T = pose_to_matrix(ref_pose[1], ref_pose[2])
        ref_T_inv = np.linalg.inv(ref_T)

        accumulated_points = []
        trajectory = []

        ts_, xyz, rpy = poses[ref_idx]
        pcd_path = os.path.join(lidar_folder, f"{ts_}.pcd")
        if not os.path.exists(pcd_path):
            continue
        points = read_pcd_with_intensity(pcd_path)

        x_old = points[:,0]
        y_old = points[:,1]
        points_new = np.copy(points)

        for i in range(0, len(poses)):
            ts, xyz, rpy = poses[i]
            T = pose_to_matrix(xyz, rpy)

            # 投影轨迹点
            traj_point = (ref_T_inv @T @ np.array([0, 0, 0, 1]))[:3]
            x_old = traj_point[0]
            y_old = traj_point[1]
            traj_point_new = np.copy(traj_point)
           
            
            trajectory.append(traj_point_new)

        trajectory_new = []
        y_anchor = trajectory[0][1]
        for i in range(len(trajectory)):
            y = trajectory[i][1]
            if y-y_anchor >2:
                trajectory_new.append(trajectory[i])
                y_anchor = y
        
        #import pdb; pdb.set_trace()

        trajectory = np.array(trajectory_new)
            

        show = False
        smoothed_traj = save_traj(trajectory, show)
        
        res = 0.16
        max_random_shift = 0.25
        trajectory_noise = np.copy(trajectory)
        smoothed_traj_noise = random_shift([trajectory_noise], max_random_shift)[0]

        # 分割数据
        trajectory_ins_noise, trajectory_ins_past_noise = split_trajectory_by_y(trajectory)
        trajectory_hmi, trajectory_hmi_past = split_trajectory_by_y(smoothed_traj_noise)

        fusion_lidar_dir = os.path.join(lidar_folder, '../local_path')
        os.makedirs(fusion_lidar_dir, exist_ok=True)
        json_path = os.path.join(fusion_lidar_dir, f"{ts_}.json")
        
        
        if len(trajectory_ins_noise)>=10 and len(trajectory_ins_past_noise)>=2 and len(trajectory_hmi)>=10 and len(trajectory_hmi_past)>=2:
            # 保存为JSON
            save_trajectory_to_json(json_path, trajectory_hmi, trajectory_hmi_past, trajectory_ins_noise, trajectory_ins_past_noise)

        #print("数据已保存到 'trajectory_data.json'")


def random_shift(traj_list, res, max_dist=1.0):
    rx = max_dist * 2 * abs(random.random() - 0.5) # [-1.0, 1.0)
    #print('rx1', rx)
    rx = rx / res
    for i in range(len(traj_list)):
        traj_list[i][:,0] += rx
    #print(rx)
    return traj_list

# 生成带有随机噪声的smoothed_traj数据
def add_noise_to_trajectory(smoothed_traj, noise_level=1.0):
    noise = np.random.uniform(-noise_level, noise_level, smoothed_traj.shape)  # 生成均匀分布噪声
    noisy_traj = smoothed_traj + noise
    return noisy_traj


def add_directional_noise_to_trajectory(smoothed_traj, noise_level=1.0, switch_ratio=0.5):

    N = smoothed_traj.shape[0]  # 轨迹点的数量
    switch_point = int(N * switch_ratio)  # 控制噪声切换的点
    
    # 为每个时间段生成噪声
    noise = np.zeros_like(smoothed_traj)
    
    # 偏左（前半段，负噪声）
    noise[:switch_point, 0] = 5*np.random.uniform(-noise_level, 0, size=switch_point)
    
    # 偏右（后半段，正噪声）
    noise[switch_point:, 0] = 5*np.random.uniform(0, noise_level, size=N - switch_point)
    
    # 对y和z轴添加随机噪声（不区分偏左偏右），模拟GPS误差
    noise[:, 1] = np.random.uniform(-noise_level, noise_level, size=N)
    noise[:, 2] = np.random.uniform(-noise_level, noise_level, size=N)
    
    # 将噪声添加到轨迹中
    noisy_traj = smoothed_traj + noise
    return noisy_traj


# 根据第二列的值分割数据
def split_trajectory_by_y(trajectory):
    trajectory_hmi = []
    trajectory_hmi_past = []
    
    for point in trajectory:
        if point[1] < 0:
            trajectory_hmi_past.append(point.tolist())  # 第二列小于0的行
        else:
            trajectory_hmi.append(point.tolist())  # 第二列大于0的行
    
    return trajectory_hmi, trajectory_hmi_past

# 生成时间戳
def generate_timestamp():
    timestamp = datetime.now()
    return {
        "sec": int(timestamp.timestamp()),
        "nsec": timestamp.microsecond * 1000  # 微秒转换为纳秒
    }

# 保存数据为JSON格式
def save_trajectory_to_json(filename, trajectory_hmi, trajectory_hmi_past, trajectory_ins, trajectory_ins_past):
    data = {
        "timestamp": generate_timestamp(),
        "curvature": -0.06956088298308505,
        "utm_pose": [
            461846.62859380859,
            5424529.11573089,
            -75.04152421146468
        ],
        "trajectory_ins": trajectory_ins,
        "trajectory_ins_past": trajectory_ins_past,
        "trajectory_hmi": trajectory_hmi,
        "trajectory_hmi_past": trajectory_hmi_past
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_traj(trajectory, show):
    # 拟合 B 样条（分别对 x, y, z 分量）
    #import pdb;pdb.set_trace()
    tck, u = splprep([trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]], s=0.5)

    # 生成均匀参数，用于重新采样轨迹
    u_fine = np.linspace(0, 1, num=500)
    smoothed = splev(u_fine, tck)
    smoothed_traj = np.vstack(smoothed).T  # shape: (500, 3)

    # 保存平滑轨迹
    #np.save(traj_path, smoothed_traj)
    print("平滑轨迹已保存为 smoothed_trajectory.npy")

    if show:
        # 可视化原始与平滑轨迹（可选）
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*trajectory.T, label='Raw', color='gray')
        ax.plot(*smoothed_traj.T, label='Smoothed', color='red')
        ax.legend()
        plt.title("B-spline Smoothed Trajectory")
        plt.show()
    
    return smoothed_traj


def process_subfolder(subfolder_path):
    # lidar_data_car_cord 路径
    lidar_folder = os.path.join(subfolder_path, "lidar_data_car_cord")
    pose_file = os.path.join(subfolder_path, "poses.txt")
    num_fusion = 50
    accumulate_pointclouds(pose_file, lidar_folder, num_fusion)


def main():
    root_folder = "./ORAD-3D-V2"
    subfolders = [os.path.join(root_folder, d) for d in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, d))]

    with Pool(processes=min(cpu_count(), len(subfolders))) as pool:
        pool.map(process_subfolder, subfolders)


# 🚀 主入口
if __name__ == "__main__":
    main()

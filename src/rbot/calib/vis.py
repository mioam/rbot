import numpy as np
import glob
import transforms3d
import cv2
from PIL import Image
from scripts.calib.utils import getXYZRGB, get_obj_in_cam
import open3d as o3d

WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])
DATE = '0110'
# left
dir = f'/home/mzc/calib/{DATE}'
# dir = '/home/mzc/1202/cali_side_r1'
intrinsic = np.load(f'{dir}/eih_intrinsic.npy')
camT = np.load(f'{dir}/eih_camT.npy')
camsideT = np.load(f'{dir}/e2h_camT.npy')
intrinsicside = np.load(f'{dir}/e2h_intrinsic.npy')

Hand = True
# Hand = False
ROOT = '/home/mzc/calib'

g_list = []
if Hand:
    dirName = DATE
    # dirName = 'cali_side2'
    flangeposList = glob.glob(f"{ROOT}/{dirName}/*.txt")
    flangeposList.sort()
    imgList = glob.glob(f"{ROOT}/{dirName}/hand/*c.png")
    imgList.sort()
    depthList = glob.glob(f"{ROOT}/{dirName}/hand/*d.png")
    depthList.sort()
    # ppp = np.array(
    #     [[-0.15788974, -0.82700796, -0.53956339,  0.83958201],
    #      [0.98733137, -0.14092512, -0.07291691, -0.31666513],
    #         [-0.01573517, -0.5442407,  0.83878154,  0.06212236],
    #         [0.,  0.,  0.,  1.],]
    # )
else:
    dirName = DATE
    flangeposList = glob.glob(f"{ROOT}/{dirName}/*.txt")
    flangeposList.sort()
    imgList = glob.glob(f"{ROOT}/{dirName}/hand/*c.png")
    imgList.sort()
    depthList = glob.glob(f"{ROOT}/{dirName}/hand/*d.png")
    depthList.sort()
    imgList2 = glob.glob(f"{ROOT}/{dirName}/side/*c.png")
    imgList2.sort()
    depthList2 = glob.glob(f"{ROOT}/{dirName}/side/*d.png")
    depthList2.sort()

for i, filename in enumerate(flangeposList):
    if len(flangeposList) > 30 and np.random.rand() < 0.9:
        continue
    ee_pose_tq = np.loadtxt(filename).tolist()
    print(ee_pose_tq)
    ee_pose_t = ee_pose_tq[:3]
    ee_pose_q = ee_pose_tq[3:]
    ee_pose_r = transforms3d.quaternions.quat2mat(ee_pose_q)

    current_pose = np.eye(4)
    current_pose[:3, :3] = ee_pose_r
    current_pose[:3, 3] = ee_pose_t
    g2w = np.linalg.inv(current_pose)

    color = cv2.imread(imgList[i])
    depth = np.array(Image.open(depthList[i]), dtype=np.float32)

    xyzrgb = getXYZRGB(color, depth, current_pose, camT, intrinsic, )
    points = xyzrgb[:, :3]
    colors = xyzrgb[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    g_list.extend([pcd, o])
    tt = get_obj_in_cam(color, intrinsic)
    obj = o3d.geometry.TriangleMesh.create_coordinate_frame(
        0.1).transform(current_pose@camT@tt)
    g_list.extend([obj])

    if not Hand:
        color2 = cv2.imread(imgList2[i])
        depth2 = np.array(Image.open(depthList2[i]), dtype=np.float32)
        xyzrgb2 = getXYZRGB(color2, depth2, np.eye(4),
                            camsideT, intrinsicside, )
        points2 = xyzrgb2[:, :3]
        colors2 = xyzrgb2[:, 3:]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        g_list.extend([pcd2])
        tt = get_obj_in_cam(color2, intrinsicside)
        if tt is None:
            continue
        obj = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.1).transform(camsideT@tt)
        g_list.extend([obj])

    # cam = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(camO)
    # c = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(ppp)
    # g_list.extend([c])

    # 添加桌面框
    points = [
        [0.3, -0.4, 0],
        [0.9, -0.4, 0],
        [0.3, 0.4, 0],
        [0.9, 0.4, 0],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    g_list.extend([line_set])

    if not Hand:
        o3d.visualization.draw_geometries(g_list)
        g_list = []

if Hand:
    o3d.visualization.draw_geometries(g_list)
    g_list = []

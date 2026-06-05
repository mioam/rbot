import cv2
import numpy as np


def _inpaint(img, missing_value=0):
    """
    pip opencv-python == 3.4.8.29
    :param image:
    :param roi: [x0,y0,x1,y1]
    :param missing_value:
    :return:
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    if scale < 1e-3:
        pdb.set_trace()
    # Has to be float32, 64 not supported.
    img = img.astype(np.float32) / scale
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale
    return img


def getXYZRGB(color, depth, robot_pose, camee_pose, camIntrinsics, inpaint=True):
    """
    Generate XYZRGB point cloud from RGB-D and poses.

    Returns:
        points: (N, 6) array [X, Y, Z, R, G, B]
    """

    H, W = depth.shape[:2]

    fx, fy, cx, cy = (
        camIntrinsics[0, 0],
        camIntrinsics[1, 1],
        camIntrinsics[0, 2],
        camIntrinsics[1, 2],
    )

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32)

    # optional depth cleanup
    if inpaint:
        depth = _inpaint(depth)
    valid = z > 0.2

    z = z[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N,4)
    T = robot_pose @ camee_pose

    pts_world = (T @ pts_cam.T).T
    pts_world = pts_world[:, :3]
    rgb = color[valid].astype(np.float32)

    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    xyzrgb = np.concatenate([pts_world, rgb], axis=1)

    return xyzrgb


def checkChessboard(color_image, shape=(11, 8)):
    color_image = color_image.copy()
    flag, corners = cv2.findChessboardCorners(
        color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    if flag:
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
    else:
        img = color_image
    return flag, img


def rodrigues_trans2tr(rvec, tvec):
    r, _ = cv2.Rodrigues(rvec)
    tvec.shape = (3,)
    T = np.identity(4)
    T[0:3, 3] = tvec
    T[0:3, 0:3] = r
    return T


pattern_size = (11, 8)
square_size = 10
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = square_size * np.mgrid[
    0 : pattern_size[0], 0 : pattern_size[1]
].T.reshape(-1, 2)
for i in range(pattern_size[0] * pattern_size[1]):
    x, y = objp[i, 0], objp[i, 1]
    objp[i, 0], objp[i, 1] = y, x


def get_obj_in_cam(color_image, mtx, shape=(11, 8)):

    color_image = color_image.copy()
    flag, corners = cv2.findChessboardCorners(
        color_image, shape, None, cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    if flag:
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        # img = cv2.drawChessboardCorners(color_image, shape, corners2, flag)
    else:
        return None
    objpoint = objp
    imgpoint = corners2

    succ, rvec, tvec = cv2.solvePnP(objpoint, imgpoint, mtx, None)

    tt = rodrigues_trans2tr(rvec, tvec / 1000.0)
    # import pdb
    # pdb.set_trace()
    return tt


def averageTransformation(H):
    """
    :param H: [h] list, h is 4*4 pose.(np.array())
    :return: average pose h
    """
    n = len(H)
    Ravg = []
    for i in range(n):
        Ravg.append(H[i][0:3, 0:3])
    R = averageRotation(Ravg)
    Tavg = 0
    for i in range(n):
        Tavg += H[i][0:3, 3]
    Havg = np.zeros((4, 4))
    Havg[0:3, 0:3] = R
    Havg[0:3, 3] = Tavg / n
    Havg[3, 3] = 1
    return Havg


def averageRotation(R):
    n = len(R)
    tmp = np.zeros((3, 3), dtype=R[0].dtype)
    for i in range(n):
        tmp = tmp + R[i]
    Rbarre = tmp / n
    RTR = np.dot(Rbarre.T, Rbarre)
    [D, V] = np.linalg.eig(RTR)
    D = np.diag(np.flipud(D))
    V = np.fliplr(V)
    sqrtD = np.sqrt(np.linalg.inv(D))
    if np.linalg.det(Rbarre[0:3, 0:3]) < 0:
        sqrtD[2, 2] = -1 * sqrtD[2, 2]
    temp = np.dot(V, sqrtD)
    temp = np.dot(temp, V.T)
    Rvag = np.dot(Rbarre, temp)
    return Rvag


def XYZRodrigues_to_Hmatrix(pos):
    """
    :param pos: [x,y,z,rodrigue]
    :return: np.array((4,4))
    """
    xyz = pos[:3]
    rod = pos[3:6]
    rot_matrix, _ = cv2.Rodrigues(np.array(rod))
    Hmatrix = np.identity(4)
    Hmatrix[:3, :3] = rot_matrix
    Hmatrix[:3, 3] = np.array(xyz).reshape(3)
    return Hmatrix


def Hmatrix_to_XYZRodrigues(Hmatrix):
    rot_matrix = Hmatrix[:3, :3]
    xyz = Hmatrix[:3, 3]
    rod, _ = cv2.Rodrigues(rot_matrix)
    return xyz.tolist() + rod.flatten().tolist()

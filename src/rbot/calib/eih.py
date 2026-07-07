from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import tyro
import yaml

from rbot.calib.utils import (
    Hmatrix_to_XYZRodrigues,
    XYZRodrigues_to_Hmatrix,
    averageTransformation,
)
from rbot.common.transformation import xyz_rot_to_mat


class Calib_extrinsic:
    def __init__(
        self, mtx, intial_pose, objpoints, imgpoints, Hm2w, hand_eye='eye_in_hand'
    ):
        # intrinsic, np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        self.mtx = mtx
        # [x,y,z,rodrigues]+[x,y,z,rodrigues], end-effector -> calib + robot base -> camera
        self.intial_pose = intial_pose
        # the 3d coordinates of cross points in calibration board frame
        self.objpoints = objpoints
        self.imgpoints = imgpoints  # uv pixels of cross points in image
        self.Hm2w = Hm2w  # robot pose
        self.hand_eye = hand_eye

    def obj_func(self, poss):
        """
        optimize the object function using least-square method
        :param pos: camera pose from robot base, [x,y,z,rodrigues]
        :return:
        """
        dist = (0.0, 0.0, 0.0, 0.0, 0.0)  # hack no distortion
        e2g = poss[0:6]  # x,y,z,rodrigues, end-effector -> calib
        b2c = poss[6:12]  # x,y,z,rodrigues, robot base -> camera
        Hc2m = XYZRodrigues_to_Hmatrix(e2g)
        Hgrid2worldAvg = XYZRodrigues_to_Hmatrix(b2c)
        mean_error_extrinsic = np.zeros(self.imgpoints[0].flatten().shape)
        mean_error_extrinsic_log = 0
        for i in range(len(self.objpoints)):
            # extrinsic param error
            temp = np.dot(self.Hm2w[i], Hc2m)
            temp = np.linalg.inv(temp)
            cam_Matrix = np.dot(temp, Hgrid2worldAvg)
            if self.hand_eye == 'eye_to_hand':
                cam_Matrix = np.linalg.inv(cam_Matrix)
            elif self.hand_eye == 'eye_in_hand':
                cam_Matrix = cam_Matrix
            cam_T = cam_Matrix[:3, 3]
            cam_T.shape = (1, 3)
            cam_R, _ = cv2.Rodrigues(cam_Matrix[:3, :3])
            imgpoints2_extrinsic, _ = cv2.projectPoints(
                self.objpoints[i], cam_R, cam_T * 1000, self.mtx, dist
            )
            error_extrinsic_logs = cv2.norm(
                self.imgpoints[i], imgpoints2_extrinsic, cv2.NORM_L2
            ) / len(imgpoints2_extrinsic)
            # pdb.set_trace()
            # error_extrinsic = abs(self.imgpoints[i] - imgpoints2_extrinsic).flatten(1)
            # mean_error_extrinsic += error_extrinsic
            mean_error_extrinsic_log += error_extrinsic_logs
        diff_log = mean_error_extrinsic_log / len(self.objpoints)
        # print(diff_log)
        # diff = mean_error_extrinsic
        return diff_log

    def run(self):
        final_pose = least_squares(self.obj_func, self.intial_pose)
        # pdb.set_trace()
        # print('--- original ---')
        # print(XYZRodrigues_to_Hmatrix(self.intial_pose[6:12]))
        # print('--- optimized ---')
        # print(XYZRodrigues_to_Hmatrix(final_pose.x[6:12]))
        return final_pose.x


class calibration:
    def __init__(self, mtx, pattern_size=(8, 6), square_size=15, handeye='EIH'):
        """

        :param image_list:  image array, num*720*1280*3
        :param pose_list: pose array, num*4*4
        :param pattern_size: calibration pattern size
        :param square_size: calibration pattern square size, 15mm
        :param handeye:
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.handeye = handeye
        self.pose_list = []
        self.joint_list = []
        self.init_calib()
        self.mtx = mtx

    def init_calib(self):
        self.objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3), np.float32
        )
        self.objp[:, :2] = self.square_size * np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        for i in range(self.pattern_size[0] * self.pattern_size[1]):
            x, y = self.objp[i, 0], self.objp[i, 1]
            self.objp[i, 0], self.objp[i, 1] = y, x
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detectFeature(self, color, show=True):
        img = color
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(
            img, self.pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH
        )  # + cv2.CALIB_CB_NORMALIZE_IMAGE+ cv2.CALIB_CB_FAST_CHECK)
        if ret == True:
            self.objpoints.append(self.objp)
            if (cv2.__version__).split('.')[0] == '2':
                cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
                corners2 = corners
            else:
                corners2 = cv2.cornerSubPix(
                    self.gray, corners, (5, 5), (-1, -1), self.criteria
                )
            self.imgpoints.append(corners2)
            if show:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                plt.title('img with feature point')
                for i in range(self.pattern_size[0] * self.pattern_size[1]):
                    ax.plot(
                        corners2[i, 0, 0],
                        corners2[i, 0, 1],
                        color='yellow',
                        marker='o',
                        markersize=12,
                    )
                plt.show()
        return ret

    def rodrigues_trans2tr(self, rvec, tvec):
        r, _ = cv2.Rodrigues(rvec)
        tvec.shape = (3,)
        T = np.identity(4)
        T[0:3, 3] = tvec
        T[0:3, 0:3] = r
        return T

    def cal(self, optimize=False):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None
        )
        self.mtx = mtx  # 这一行是加的，希望以新校准的intr为准。后面的代码其实混用了 self.mtx 和 mtx，不知道有没有精度问题
        print(mtx)
        Hg2c = []
        pose_list = np.array(self.pose_list)
        for i in range(len(rvecs)):
            tt = self.rodrigues_trans2tr(rvecs[i], tvecs[i] / 1000.0)
            Hg2c.append(tt)  # target to camera
        Hg2c = np.array(Hg2c)

        print(pose_list.shape, Hg2c.shape)

        rot, pos = cv2.calibrateHandEye(
            pose_list[:, :3, :3], pose_list[:, :3, 3], Hg2c[:, :3, :3], Hg2c[:, :3, 3]
        )
        camT = np.identity(4)
        camT[:3, :3] = rot
        camT[:3, 3] = pos[:, 0]
        Hc2m = camT
        print('===========================camT===========================')
        print(camT)
        if optimize:
            Hc2w = []
            Hw2c = []
            Hg2w = []
            Hw2g = []
            for i in range(len(rvecs)):
                temp = np.dot(pose_list[i], Hc2m)
                Hc2w.append(temp)
                Hw2c.append(np.linalg.inv(temp))
                temp = np.dot(temp, Hg2c[i])
                Hg2w.append(temp)
                Hw2g.append(np.linalg.inv(temp))

            Hgrid2worldAvg = averageTransformation(Hg2w)
            mean_error = 0
            mean_error_extrinsic = 0
            # pdb.set_trace()
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                    imgpoints2
                )
                mean_error += error
                # pdb.set_trace()
                # extrisick param error
                temp = np.dot(pose_list[i], Hc2m)
                temp = np.linalg.inv(temp)
                cam_Matrix = np.dot(temp, Hgrid2worldAvg)
                cam_Matrix = cam_Matrix
                cam_T = cam_Matrix[:3, 3]
                cam_T.shape = (3, 1)
                cam_R, _ = cv2.Rodrigues(cam_Matrix[:3, :3])
                # pdb.set_trace()
                imgpoints2_extrinsic, _ = cv2.projectPoints(
                    self.objpoints[i], cam_R, cam_T * 1000, mtx, dist
                )
                error_extrinsic = cv2.norm(
                    self.imgpoints[i], imgpoints2_extrinsic, cv2.NORM_L2
                ) / len(imgpoints2_extrinsic)
                mean_error_extrinsic += error_extrinsic
                # pdb.set_trace()
            e2g = Hmatrix_to_XYZRodrigues(Hc2m)
            b2c = Hmatrix_to_XYZRodrigues(Hgrid2worldAvg)
            # pdb.set_trace()
            intial_pose = e2g + b2c
            cali = Calib_extrinsic(
                self.mtx,
                intial_pose,
                self.objpoints,
                self.imgpoints,
                pose_list,
                hand_eye='eye_in_hand',
            )
            final_pose = cali.run()
            for i in range(3):
                intial_pose = final_pose
                cali = Calib_extrinsic(
                    self.mtx,
                    intial_pose,
                    self.objpoints,
                    self.imgpoints,
                    pose_list,
                    hand_eye='eye_in_hand',
                )
                final_pose = cali.run()
            Hc2m = XYZRodrigues_to_Hmatrix(final_pose[:6])
            camT = Hc2m
        return camT


def main(DIR: Path):
    with (DIR / 'config.yaml').open('r') as f:
        conf = yaml.load(f, yaml.BaseLoader)
    print(conf)
    cam_hand = conf['cam_hand']
    cam_side = conf['cam_side']
    intr = np.load(DIR / f'{cam_hand}_intr.npy')
    calib = calibration(intr, (11, 8), 25)
    timestamps = [p.stem for p in (DIR / 'pose').glob('*.txt')]
    timestamps.sort()
    print(len(timestamps))

    for i, timestamp in enumerate(timestamps):
        pose_filename = DIR / 'pose' / f'{timestamp}.txt'
        img_filename = DIR / f'{cam_hand}' / f'{timestamp}c.png'
        if not img_filename.exists():
            print(' [INFO] No image ', img_filename)
            continue

        ee_pose_tq = np.loadtxt(pose_filename).tolist()
        current_pose = xyz_rot_to_mat(ee_pose_tq, from_rep='quaternion')

        color = cv2.imread(img_filename)

        flag = calib.detectFeature(color, show=False)
        if flag:
            calib.pose_list.append(current_pose)
        else:
            print(' [INFO] No chessboard detected in image ', img_filename)

    print(len(calib.objpoints))
    camT = calib.cal(optimize=True)
    print('=' * 10)
    print(camT)

    (DIR / 'calib').mkdir(exist_ok=True)
    np.save(DIR / 'calib' / f'{cam_hand}_intr.npy', calib.mtx)
    np.save(DIR / 'calib' / f'{cam_hand}_camT.npy', camT)


if __name__ == '__main__':
    tyro.cli(main)
    # np.save(os.path.join(f'{dirName}/', 'eih_intrinsic.npy'), calib.mtx)
    # np.save(os.path.join(f'{dirName}/', 'eih_camT.npy'), camT)
    # np.save('config/franka_campose.npy',camT)

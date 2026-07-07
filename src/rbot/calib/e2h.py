from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import tyro
import yaml

from rbot.calib.utils import (
    XYZRodrigues_to_Hmatrix,
)
from rbot.common.transformation import (
    rotation_distance,
    xyz_rot_to_mat,
)

# from flexiv import FlexivRobot


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
            # corners2 = corners
            if (cv2.__version__).split('.')[0] == '2':
                # pdb.set_trace()
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

    def get_Hg2c(self):
        # import pdb;pdb.set_trace()
        if True:  # 固定mtx
            rvecs, tvecs = [], []
            for i in range(len(self.objpoints)):
                succ, rvec, tvec = cv2.solvePnP(
                    self.objpoints[i], self.imgpoints[i], self.mtx, None
                )
                rvecs.append(rvec)
                tvecs.append(tvec)
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None
            )
        Hg2c = []
        for i in range(len(rvecs)):
            tt = self.rodrigues_trans2tr(rvecs[i], tvecs[i] / 1000.0)
            Hg2c.append(tt)
        Hg2c = np.array(Hg2c)
        return Hg2c

    def cal(self, calib_wrist, camera_pose):
        def transformation_error_func(params, obj2base, obj2cam):
            tx, ty, tz, qw, qx, qy, qz = params
            estimated_transform = xyz_rot_to_mat(
                np.array(params, dtype=np.float64),
                from_rep='quaternion',
            )
            # 注意这里一定要有float64的精度

            error = []

            for i in range(len(obj2base)):
                # Compute the error between the observed transformation and the one computed by the estimated parameters
                observed_transform = obj2base[i] @ np.linalg.inv(obj2cam[i])
                diff_transform = observed_transform @ np.linalg.inv(estimated_transform)

                # Calculate translational error
                translational_error = np.linalg.norm(diff_transform[:3, 3])

                # Calculate rotational error (as an angle)
                angle = rotation_distance(diff_transform[:3, :3])
                # _, angle = transforms3d.axangles.mat2axangle(diff_transform[:3, :3])

                # Accumulate errors
                error.append(translational_error)
                error.append(angle)

            return error

        # camera_pose = np.array([[ 0.00886583,  0.99944172,  0.03221257, -0.07624523],
        #                         [-0.99975617,  0.00951093, -0.01992849,  0.06389011],
        #                         [-0.02022373, -0.03202803,  0.99928235,  0.00224033],
        #                         [ 0.,  0.,  0.,  1.]])
        # camera_pose = np.array(
        # [[ 0.03089147, 0.99886808, -0.03617001, -0.05927289],
        # [-0.99915533, 0.02987879, -0.02821134,  0.02904617],
        # [-0.02709869, 0.03701095,  0.99894737, -0.20291156],
        # [ 0.        , 0.        ,  0.        ,  1.        ],]
        # )

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None)
        mtx = self.mtx.copy()
        print('intrinsic', mtx)
        self.mtx = mtx

        pose_list = np.array(self.pose_list)
        Hg2c = calib_wrist.get_Hg2c()
        # obj2base = pose_list[0] @ camera_pose @ Hg2c
        obj2base = pose_list @ camera_pose @ Hg2c
        # import pdb;pdb.set_trace()
        obj2cam = self.get_Hg2c()
        # print(obj2base)

        camT = []
        for i in range(len(pose_list)):
            ppp = (obj2base[i]) @ np.linalg.inv(obj2cam[i])
            camT.append(ppp)
            print(ppp)
            # p.addUserDebugLine(
            #     ppp[:3, 3], ppp[:3, 3] + ppp[0, :3] * 0.1, (1, 0, 0), 3.0
            # )
            # p.addUserDebugLine(
            #     ppp[:3, 3], ppp[:3, 3] + ppp[1, :3] * 0.1, (0, 1, 0), 3.0
            # )
            # p.addUserDebugLine(
            #     ppp[:3, 3], ppp[:3, 3] + ppp[2, :3] * 0.1, (0, 0, 1), 3.0
            # )
            # input(f'{i}')

        # Assume no initial rotation and translation
        # initial_guess = [0, 0, 0, 0, 0, 0, 1]
        initial_guess = [0, 0, 0, 1, 0, 0, 0]

        print(obj2base.shape, initial_guess)
        res = least_squares(
            transformation_error_func, initial_guess, args=(obj2base, obj2cam)
        )

        camT = xyz_rot_to_mat(
            [
                res.x[0],
                res.x[1],
                res.x[2],
                res.x[3],
                res.x[4],
                res.x[5],
                res.x[6],
            ],
            from_rep='quaternion',
        )

        # optimized_quaternion = [
        #     res.x[6],
        #     res.x[3],
        #     res.x[4],
        #     res.x[5],
        # ]  # [qw, qx, qy, qz]
        # optimized_rotation = transforms3d.quaternions.quat2mat(optimized_quaternion)
        # optimized_translation = res.x[:3]

        # optimized_cam2_to_base_transform = np.vstack((optimized_rotation, [0, 0, 0]))
        # optimized_cam2_to_base_transform = np.hstack((
        #     optimized_cam2_to_base_transform,
        #     [
        #         [optimized_translation[0]],
        #         [optimized_translation[1]],
        #         [optimized_translation[2]],
        #         [1],
        #     ],
        # ))

        # camT = optimized_cam2_to_base_transform
        # camT = np.mean(camT, axis=0)
        return camT


def main(DIR: Path):
    with (DIR / 'config.yaml').open('r') as f:
        conf = yaml.load(f, yaml.BaseLoader)
    print(conf)
    cam_hand = conf['cam_hand']
    cam_sides = conf['cam_side']
    for cam_side in cam_sides:
        run(DIR, cam_hand, cam_side, (11, 8), 25)


def run(DIR: Path, cam_hand, cam_side, pattern_size, square_size):

    intr_wrist = np.load(DIR / 'calib' / f'{cam_hand}_intr.npy')
    camera_pose = np.load(DIR / 'calib' / f'{cam_hand}_camT.npy')

    intr = np.load(DIR / f'{cam_side}_intr.npy')
    calib = calibration(intr, pattern_size, square_size)
    calib_wrist = calibration(intr_wrist, pattern_size, square_size)
    timestamps = [p.stem for p in (DIR / 'pose').glob('*.txt')]
    timestamps.sort()
    print(len(timestamps))

    for timestamp in timestamps:
        pose_filename = DIR / 'pose' / f'{timestamp}.txt'
        hand_filename = DIR / f'{cam_hand}' / f'{timestamp}c.png'
        side_filename = DIR / f'{cam_side}' / f'{timestamp}c.png'
        if not hand_filename.exists():
            print(' [INFO] No image ', hand_filename)
            continue
        if not side_filename.exists():
            print(' [INFO] No image ', side_filename)
            continue

        ee_pose_tq = np.loadtxt(pose_filename).tolist()
        current_pose = xyz_rot_to_mat(ee_pose_tq, from_rep='quaternion')

        color = cv2.imread(side_filename)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color_ = cv2.imread(hand_filename)
        color_ = cv2.cvtColor(color_, cv2.COLOR_RGB2BGR)
        # cv2.imshow('a', color_)
        # cv2.waitKey(10)
        flag, _ = cv2.findChessboardCorners(
            color, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        flag_, _ = cv2.findChessboardCorners(
            color_, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH
        )

        if flag and flag_:
            flag = calib.detectFeature(color, show=False)
            flag_ = calib_wrist.detectFeature(color_, show=False)
            calib.pose_list.append(current_pose)

        else:
            print(flag, flag_)
            print(
                ' [INFO] No chessboard detected in image ', hand_filename, side_filename
            )

    print(len(calib.objpoints))
    camT = calib.cal(calib_wrist, camera_pose)
    print('=' * 10)
    print(camT)

    (DIR / 'calib').mkdir(exist_ok=True)
    np.save(DIR / 'calib' / f'{cam_side}_intr.npy', calib.mtx)
    np.save(DIR / 'calib' / f'{cam_side}_camT.npy', camT)


if __name__ == '__main__':
    tyro.cli(main)

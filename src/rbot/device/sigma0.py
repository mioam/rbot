# fmt: off
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "sigma_sdk")
import sigma7

# fmt: on


def parse(px, py, pz, oa, ob, og):
    # return np.array([py, -px, pz]), np.array([oa, ob, og])
    # return np.array([px, py, pz]), R.from_euler('xyz', np.array([-og, ob, -oa]), degrees=False)
    return np.array([px, py, pz]), R.from_euler(
        'XYZ', np.array([oa, ob, og]), degrees=False
    )
    # return np.array([-px, -py, pz]), np.array([ob, -oa, og])


class Sigma7:
    def __init__(self, pos_scale=5, width_scale=1000) -> None:
        self.pos_scale = pos_scale
        self.width_scale = width_scale
        self.start_sigma()
        self.init_p, self.init_r, _ = self.get_current()
        self.detached = False

    def start_sigma(self):
        sigma7.drdOpen()
        sigma7.drdAutoInit()
        print('starting sigma')
        sigma7.drdRegulatePos(on=False)
        sigma7.drdRegulateRot(on=False)
        sigma7.drdRegulateGrip(on=False)
        sigma7.drdStart()
        print('sigma ready')

    def get_current(self):
        sig, px, py, pz, oa, ob, og, pg, matrix = sigma7.drdGetPositionAndOrientation()
        # x，y, z，绕x旋转，绕y旋转，绕z旋转
        curr_p, curr_r = parse(px, py, pz, oa, ob, og)
        print(sig, sigma7.dhdGetStatus())
        return curr_p, curr_r, pg

    def get_control(self):
        curr_p, curr_r, pg = self.get_current()

        diff_p = curr_p - self.init_p
        diff_r = curr_r * self.init_r.inv()
        diff_p = diff_p * self.pos_scale
        width = pg / -0.027 * self.width_scale
        return diff_p, diff_r, width

    def detach(self):
        if not self.detached:
            self._prev_p, self._prev_r, _ = self.get_current()
            self.detached = True

    def detach_init(self):
        self.init_p = self._prev_p
        self.init_r = self._prev_r

    def resume(self):
        if self.detached:
            curr_p, curr_r, _ = self.get_current()
            self.init_p = self.init_p - self._prev_p + curr_p
            self.init_r = self.init_r * self._prev_r.inv() * curr_r
            self.detached = False


if __name__ == '__main__':
    import time

    sigma = Sigma7()
    time.sleep(1)
    while True:
        sigma.get_control()
        time.sleep(1)

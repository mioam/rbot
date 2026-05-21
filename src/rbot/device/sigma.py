import atexit
import contextlib
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse(px, py, pz, oa, ob, og):
    return np.array([px, py, pz]), R.from_euler(
        'XYZ', np.array([oa, ob, og]), degrees=False
    )


def _recv_timeout(conn, timeout=1.0):
    if not conn.poll(timeout):
        raise TimeoutError
    return conn.recv()


class SigmaProcess:
    def __init__(self, pipe):
        self.pipe = pipe

    def run(self):
        available = sorted(os.sched_getaffinity(0))
        nonzero = [c for c in available if c != 0]
        if nonzero:
            target = {nonzero[-1]}
            os.sched_setaffinity(0, target)
        # 仅在子进程中导入和初始化 SDK，实现彻底隔离
        sys.path.insert(0, 'sigma_sdk')
        import sigma7

        sigma7.drdOpen()
        sigma7.drdAutoInit()
        sigma7.drdRegulatePos(on=False)
        sigma7.drdRegulateRot(on=False)
        sigma7.drdRegulateGrip(on=False)
        sigma7.drdStart()

        print('[Sigma SDK] Backend driver started on separate core.')
        self.pipe.send(('READY',))

        try:
            while True:
                # 阻塞等待主进程指令
                if self.pipe.poll(0.1):
                    cmd = self.pipe.recv()
                    if cmd == 'GET_POS':
                        data = sigma7.drdGetPositionAndOrientation()
                        self.pipe.send(('OK', data))
                    elif cmd == 'CLOSE':
                        self.pipe.send(('OK',))
                        break
        finally:
            sigma7.drdClose()
            self.pipe.close()


class Sigma7:
    def __init__(self, pos_scale=5, width_scale=1000) -> None:
        self.pos_scale = pos_scale
        self.width_scale = width_scale

        # 启动独立进程
        self._ctx = mp.get_context('spawn')
        self._parent_conn, self._child_conn = self._ctx.Pipe()
        self.hw_handler = SigmaProcess(self._child_conn)
        self._process = self._ctx.Process(target=self.hw_handler.run, daemon=False)
        self._process.start()
        atexit.register(self.close)
        self._closed = False
        msg = self._parent_conn.recv()
        assert msg[0] == 'READY'

        self.init_p, self.init_r, _ = self.get_current()
        self.detached = False

    def get_current(self):
        """通过 Pipe 向子进程请求数据"""
        self._parent_conn.send('GET_POS')
        ok, res = self._parent_conn.recv()
        assert ok == 'OK'
        sig, px, py, pz, oa, ob, og, pg = res[0:8]
        curr_p, curr_r = parse(px, py, pz, oa, ob, og)
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

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._parent_conn.send('CLOSE')
                self._parent_conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            with contextlib.suppress(OSError):
                self._parent_conn.close()
            if self._process.is_alive():
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

    def __del__(self):
        self.close()


if __name__ == '__main__':
    import time

    sigma = Sigma7()
    time.sleep(1)
    while True:
        sigma.get_control()
        time.sleep(1)

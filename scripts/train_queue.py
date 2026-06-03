from datetime import datetime
from multiprocessing import Lock, Process, Queue
import os
from pathlib import Path
import subprocess
import sys
import time

import pynvml

AVAILABLE_GPUS = [1, 2, 3, 5, 6, 7]
GPUS_PER_TASK = 2
LOG_DIR = Path('logs')  # 存放每个实验日志的目录
EXP_NAME = f'exp-{datetime.now().strftime("%y%m%d-%H%M%S")}'

TASKS = [
    # f'uv run scripts/compute_norm_stats.py --model.name {base} '
    # f'--train.repo_id {repo_id} --data.state {state} --data.action {action} --data.delta {delta} --exp-name {EXP_NAME}'
    f'XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py --model.name {base} '
    f'--train.repo_id {repo_id} --data.state {state} --data.action {action} --data.delta {delta} --exp-name {EXP_NAME}'
    for base in ['pi0', 'pi05']
    for repo_id in ['miaom/carrot_fix_pot']
    for state in ['rg', 'qg', 'eg']
    for action in ['r', 'q', 'e']
    for delta in [True, False]
]

print_lock = Lock()
file_lock = Lock()


def log_status(msg):
    """安全的终端状态打印"""
    with print_lock:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}] {msg}')


def check_gpus_free(gpu_ids, threshold=0.05):
    pynvml.nvmlInit()
    for gpu_id in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = info.used / info.total
        if usage > threshold:
            return False, f'GPU {gpu_id} 显存占用过高 ({usage * 100:.1f}%)'
    return True, 'OK'


def worker(task_queue, gpu_pool_queue, total_tasks):
    while True:
        item = task_queue.get()
        if item is None:
            break

        task_idx, task = item

        while True:
            gpus = gpu_pool_queue.get()
            gpu_str = ','.join(map(str, gpus))

            is_free, reason = check_gpus_free(gpus, threshold=0.05)
            if is_free:
                break
            log_status(
                f'[等待] 任务:[{task_idx}] | 显卡:[{gpu_str}] 暂不可用: {reason}。30秒后重新检测...'
            )
            gpu_pool_queue.put(gpus)
            time.sleep(30)

        exp_dir = Path(LOG_DIR) / EXP_NAME
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = exp_dir / f'{task_idx}.log'

        log_status(f'[启动] 进度:[{task_idx}/{total_tasks}] | 显卡:[{gpu_str}]')

        current_env = os.environ.copy()
        current_env['CUDA_VISIBLE_DEVICES'] = gpu_str

        with log_file_path.open('w', encoding='utf-8') as f:
            f.write(
                f'=== TASK COMMAND ===\nCUDA_VISIBLE_DEVICES={gpu_str} {task}\n====================\n\n'
            )

        start_time = time.time()
        try:
            with log_file_path.open('a', encoding='utf-8') as f:
                process = subprocess.Popen(
                    task,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=current_env,
                    bufsize=0,
                )
                process.wait()

            elapsed = time.time() - start_time
            if process.returncode == 0:
                log_status(
                    f'[成功] 进度:[{task_idx}/{total_tasks}] | 显卡:[{gpu_str}] | 耗时: {elapsed:.1f}s'
                )
            else:
                raise subprocess.CalledProcessError(process.returncode, task)
        except Exception as e:
            elapsed = time.time() - start_time
            log_status(
                f'[失败] 进度:[{task_idx}/{total_tasks}] | 显卡:[{gpu_str}] | 异常退出 | 耗时: {elapsed:.1f}s'
            )

            # 使用 file_lock 保证多进程写同一个错误日志时不会错位交织
            with file_lock:
                failed_log_path = Path(LOG_DIR) / 'failed_tasks.log'
                with failed_log_path.open('a', encoding='utf-8') as err_f:
                    err_f.write(
                        f'[{datetime.now()}] task: {task_idx} | GPUs: {gpu_str} | Cmd: {task} | Code: {getattr(e, "returncode", "Unknown")}\n'
                    )

        finally:
            gpu_pool_queue.put(gpus)


if __name__ == '__main__':
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_tasks = len(TASKS)
    if total_tasks == 0:
        print('没有检测到待训练的任务')
        sys.exit(0)

    # 3. 显卡分组
    gpu_groups = [
        tuple(AVAILABLE_GPUS[i : i + GPUS_PER_TASK])
        for i in range(0, len(AVAILABLE_GPUS), GPUS_PER_TASK)
    ]
    max_workers = len(gpu_groups)

    print('=' * 60)
    print(f' 多卡训练队列服务启动 | 总任务数: {total_tasks}')
    print(f' 显卡并行分组: {gpu_groups} (最大并行数: {max_workers})')
    print(f' 实验日志: {LOG_DIR}/{EXP_NAME}/<task_idx>.log')
    print('=' * 60)

    task_queue = Queue()
    gpu_pool_queue = Queue()

    for idx, task in enumerate(TASKS):
        task_queue.put((idx, task))
    for group in gpu_groups:
        gpu_pool_queue.put(group)
    for _ in range(max_workers):
        task_queue.put(None)

    processes = []
    for i in range(max_workers):
        p = Process(target=worker, args=(task_queue, gpu_pool_queue, total_tasks))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('\n所有任务已处理完毕！')
    failed_log_path = LOG_DIR / 'failed_tasks.log'
    if os.path.exists(failed_log_path):
        print(f' 警告：部分实验运行失败，请查看 {failed_log_path}')

# ai generated
import os
from pathlib import Path
import shutil
import subprocess


def is_all_png(files):
    return len(files) > 0 and all(f.lower().endswith('.png') for f in files)


def get_dir_size(files, folder):
    return sum((folder / f).stat().st_size for f in files)


def make_video_from_images(folder: Path, files, output_path):
    temp_dir = folder / '_temp_seq'
    temp_dir.mkdir(exist_ok=True)

    files = sorted(files)

    for i, f in enumerate(files):
        src = folder / f
        dst = temp_dir / f'{i:06d}.png'
        shutil.copy2(src, dst)

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        '24',
        '-i',
        str(temp_dir / '%06d.png'),
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        str(output_path),
    ]

    subprocess.run(cmd, check=True)

    shutil.rmtree(temp_dir)


def process_folder(folder: Path, delete_images=False):
    files = [f.name for f in folder.iterdir() if f.is_file()]

    if len(files) <= 20:
        return

    if not is_all_png(files):
        return

    print(f'处理目录: {folder}')

    files_sorted = sorted(files)

    original_size = get_dir_size(files_sorted, folder)
    video_path = folder / f'{folder.name}.mp4'

    try:
        make_video_from_images(folder, files_sorted, video_path)
    except subprocess.CalledProcessError:
        print(f'ffmpeg失败: {folder}')
        return

    video_size = video_path.stat().st_size
    ratio = original_size / video_size if video_size > 0 else 0

    print(f'原图大小: {original_size / 1024:.2f} KB')
    print(f'视频大小: {video_size / 1024:.2f} KB')
    print(f'压缩比: {ratio:.2f}x')

    if delete_images:
        for f in files_sorted:
            (folder / f).unlink()
        print('已删除原图')


def process_directory(root_dir, delete_images=False):
    root = Path(root_dir)

    # 递归遍历所有目录
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        process_folder(folder, delete_images)


if __name__ == '__main__':
    target_dir = '/ssd1/mzc/workspace/Knowledge-IL/test'  # 改成你的路径
    process_directory(target_dir, delete_images=True)

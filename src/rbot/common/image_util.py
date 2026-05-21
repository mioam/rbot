from PIL import Image

def center_crop_resize(image: Image.Image, target_size: tuple):
    target_width, target_height = target_size
    original_width, original_height = image.size

    # 计算裁剪区域比例
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_width = int(original_height * target_ratio)
        left = (original_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = original_height
    else:
        new_height = int(original_width / target_ratio)
        left = 0
        top = (original_height - new_height) // 2
        right = original_width
        bottom = top + new_height

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(target_size, Image.LANCZOS)

    return resized_image
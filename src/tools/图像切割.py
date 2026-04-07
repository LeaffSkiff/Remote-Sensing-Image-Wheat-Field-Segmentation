from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None  # 关闭最大限制

def split_image(image_path, output_dir, tile_size=512, overlap=0):
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    w, h = img.size

    stride = tile_size - overlap
    tiles = []

    # 先收集所有 tile 位置，方便统计总数
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            tiles.append((left, top))

    total_tiles = len(tiles)
    num_digits = len(str(total_tiles))  # 计算需要几位数字（比如总共123张，那就用3位）

    for count, (left, top) in enumerate(tiles):
        right = min(left + tile_size, w)
        bottom = min(top + tile_size, h)
        box = (left, top, right, bottom)
        tile = img.crop(box)
        filename = f"tile_{count + 1:0{num_digits}d}.png"  # 加前导零
        tile.save(os.path.join(output_dir, filename))

# 示例调用
split_image("../Resource/TrainData/0.jpg", "../MidPicture", tile_size=1024, overlap=50)

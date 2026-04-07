import os
import re

# 设置你的目标目录
TARGET_DIR = '../MidPicture'  # Windows 示例

pattern = re.compile(r"title_(\d+)\.(png|json)")

# 分别收集 png 和 json
png_files = []
json_files = []

for filename in os.listdir(TARGET_DIR):
    match = pattern.match(filename)
    if match:
        index = int(match.group(1))
        ext = match.group(2)
        if ext == "png":
            png_files.append((index, filename))
        elif ext == "json":
            json_files.append((index, filename))

# 分别排序
png_files.sort()
json_files.sort()

# 重命名 PNG 文件
for i, (_, old_filename) in enumerate(png_files, 1):
    new_name = f"{i:03d}.png"
    old_path = os.path.join(TARGET_DIR, old_filename)
    new_path = os.path.join(TARGET_DIR, new_name)

    if os.path.exists(new_path):
        print(f"[PNG] 跳过：{new_name} 已存在")
        continue

    os.rename(old_path, new_path)
    print(f"[PNG] 重命名：{old_filename} -> {new_name}")

# 重命名 JSON 文件
for i, (_, old_filename) in enumerate(json_files, 1):
    new_name = f"{i:03d}.json"
    old_path = os.path.join(TARGET_DIR, old_filename)
    new_path = os.path.join(TARGET_DIR, new_name)

    if os.path.exists(new_path):
        print(f"[JSON] 跳过：{new_name} 已存在")
        continue

    os.rename(old_path, new_path)
    print(f"[JSON] 重命名：{old_filename} -> {new_name}")

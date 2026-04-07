import os
import subprocess

json_dir = r"D:\Develop\PyCharm\Unet\MidPicture"
output_dir = r"D:\Develop\PyCharm\Unet\labelme_output_4"

os.makedirs(output_dir, exist_ok=True)

python_exe = r"C:\Users\28027\miniconda3\envs\labelme\python.exe"

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        out_folder = os.path.join(output_dir, filename.replace(".json", ""))
        cmd = [
            python_exe, "-m", "labelme.cli.json_to_dataset",
            json_path, "-o", out_folder
        ]
        subprocess.run(cmd)

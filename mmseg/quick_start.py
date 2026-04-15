"""
快速启动脚本 - 不需要进入 mmsegmentation 目录

使用方法:
    python quick_start.py train   # 开始训练
    python quick_start.py prepare # 准备数据
"""
import subprocess
import sys
from pathlib import Path

# ============ 配置区域 ============
# MMSegmentation 安装路径 (conda 环境)
MMSEG_ROOT = Path(r'C:\Users\28027\miniconda3\envs\mmseg_env\Lib\site-packages\mmseg')

# 配置文件选择
CONFIG = 'mmseg/configs/wheat/wheat_deeplabv3_r50.py'  # 或 wheat_unet_r50.py

# 工作目录 (模型和日志输出位置)
WORK_DIR = 'mmseg/work_dirs/wheat_deeplabv3'

# GPU 设置
GPU_ID = '0'  # 使用哪块 GPU
# ============ 配置区域结束 ============

def check_env():
    """检查环境"""
    print("检查 MMSegmentation 环境...")
    try:
        import mmseg
        print(f"MMSegmentation 版本：{mmseg.__version__}")

        import torch
        print(f"PyTorch 版本：{torch.__version__}")
        print(f"CUDA 可用：{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        return True
    except ImportError as e:
        print(f"环境检查失败：{e}")
        print("请确保已激活正确的 conda 环境")
        return False

def prepare_data():
    """准备数据集"""
    print("\n准备数据集...")
    from src.tools.prepare_mmseg_data import prepare_data
    prepare_data()
    print("数据集准备完成！")

def train():
    """开始训练"""
    print(f"\n开始训练...")
    print(f"配置文件：{CONFIG}")
    print(f"工作目录：{WORK_DIR}")
    print(f"GPU: {GPU_ID}")

    # 构建命令
    cmd = [
        sys.executable,
        str(MMSEG_ROOT / 'tools' / 'train.py'),
        CONFIG,
        '--work-dir', WORK_DIR,
        '--device', f'cuda:{GPU_ID}'
    ]

    print(f"\n执行命令：{' '.join(cmd)}")
    subprocess.run(cmd, cwd=MMSEG_ROOT)

def evaluate(checkpoint):
    """评估模型"""
    print(f"\n评估模型：{checkpoint}")

    cmd = [
        sys.executable,
        str(MMSEG_ROOT / 'tools' / 'test.py'),
        CONFIG,
        checkpoint,
        '--eval', 'mIoU',
        '--device', f'cuda:{GPU_ID}'
    ]

    subprocess.run(cmd, cwd=MMSEG_ROOT)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'prepare':
        prepare_data()
    elif command == 'train':
        if not check_env():
            sys.exit(1)
        train()
    elif command == 'eval':
        if len(sys.argv) < 3:
            print("用法：python quick_start.py eval <checkpoint_path>")
            sys.exit(1)
        evaluate(sys.argv[2])
    else:
        print(f"未知命令：{command}")
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()

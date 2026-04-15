@echo off
chcp 65001 >nul
echo ========================================
echo   小麦田分割 MMSegmentation 训练脚本
echo ========================================
echo.

set CONDA_ENV=C:\Users\28027\miniconda3\envs\mmseg_env
set PYTHON=%CONDA_ENV%\python.exe
set MMSEG_ROOT=%CONDA_ENV%\Lib\site-packages\mmseg

if not exist "%PYTHON%" (
    echo [错误] 找不到 Python: %PYTHON%
    echo 请检查 conda 环境 mmseg_env 是否存在
    pause
    exit /b 1
)

echo [信息] 使用 Python: %PYTHON%
echo [信息] MMSegmentation: %MMSEG_ROOT%
echo.

:menu
echo 请选择操作:
echo   1. 准备数据集
echo   2. 开始训练 (DeepLabV3+)
echo   3. 开始训练 (UNet)
echo   4. 评估模型
echo   5. 预测图像
echo   0. 退出
echo.
set /p choice=请输入选项 (0-5):

if "%choice%"=="1" goto prepare
if "%choice%"=="2" goto train_deeplab
if "%choice%"=="3" goto train_unet
if "%choice%"=="4" goto eval
if "%choice%"=="5" goto predict
if "%choice%"=="0" goto end
goto menu

:prepare
echo.
echo [准备数据集...]
"%PYTHON%" mmseg\tools\prepare_mmseg_data.py
echo 完成!
pause
goto menu

:train_deeplab
echo.
echo [开始训练 DeepLabV3+...]
"%PYTHON%" "%MMSEG_ROOT%\tools\train.py" mmseg\configs\wheat\wheat_deeplabv3_r50.py --work-dir mmseg\work_dirs\wheat_deeplabv3
pause
goto menu

:train_unet
echo.
echo [开始训练 UNet...]
"%PYTHON%" "%MMSEG_ROOT%\tools\train.py" mmseg\configs\wheat\wheat_unet_r50.py --work-dir mmseg\work_dirs\wheat_unet
pause
goto menu

:eval
echo.
set /p ckpt=请输入模型权重路径 (例如 work_dirs\wheat_unet\iter_40000.pth):
"%PYTHON%" "%MMSEG_ROOT%\tools\test.py" mmseg\configs\wheat\wheat_unet_r50.py "%ckpt%" --eval mIoU
pause
goto menu

:predict
echo.
set /p img=请输入图像路径:
"%PYTHON%" mmseg\tools\predict_mmseg.py --config mmseg\configs\wheat\wheat_unet_r50.py --checkpoint mmseg\work_dirs\wheat_unet\iter_40000.pth --image "%img%" --output predictions
pause
goto menu

:end
echo 再见!

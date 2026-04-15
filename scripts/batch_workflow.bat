@echo off
title Semi-automatic Annotation Workflow
chcp 65001 >nul

echo ========================================
echo   Semi-automatic Annotation Workflow
echo ========================================
echo.

pause

set PYTHON=C:\Users\28027\miniconda3\envs\mmseg_env\python.exe

echo Checking Python: %PYTHON%
if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    pause
    exit /b 1
)
echo OK
echo.

:menu
echo.
echo Select operation:
echo   1. Model Prediction (generate masks)
echo   2. Convert to labelme format (for editing)
echo   3. Export labelme to training data
echo   4. Full workflow (1 then 2)
echo   0. Exit
echo.
set /p choice=Enter choice (0-4):

if "%choice%"=="1" goto predict
if "%choice%"=="2" goto to_labelme
if "%choice%"=="3" goto to_mmseg
if "%choice%"=="4" goto full
if "%choice%"=="0" goto end
goto menu

:predict
echo.
echo === Model Prediction ===
set /p ckpt=Model checkpoint path (default=mmseg/work_dirs/wheat_unet/iter_40000.pth):
if "%ckpt%"=="" set ckpt=mmseg\work_dirs\wheat_unet\iter_40000.pth

set /p imgdir=Input image dir (default=to_predict):
if "%imgdir%"=="" set imgdir=to_predict

echo Running...
"%PYTHON%" mmseg\tools\predict_mmseg.py --config mmseg\configs\wheat\wheat_unet_r50.py --checkpoint "%ckpt%" --image "%imgdir%" --output predictions
echo Done! Check predictions/ folder
pause
goto menu

:to_labelme
echo.
echo === Convert to labelme format ===
set /p preddir=Prediction mask dir (default=predictions):
if "%preddir%"=="" set preddir=predictions

set /p imgdir=Original image dir (default=to_predict):
if "%imgdir%"=="" set imgdir=to_predict

echo Running...
"%PYTHON%" mmseg\tools\pred_to_labelme.py --pred "%preddir%" --image "%imgdir%" --output data_set\labelme_annotations
echo Done! Open data_set\labelme_annotations in labelme to edit
pause
goto menu

:to_mmseg
echo.
echo === Export labelme to training data ===
set /p lbdir=labelme JSON directory:
set /p outimg=Output image dir (default=data\processed\images\train):
if "%outimg%"=="" set outimg=data\processed\images\train

set /p outmask=Output mask dir (default=data\processed\annotations\train):
if "%outmask%"=="" set outmask=data\processed\annotations\train

echo Running...
"%PYTHON%" mmseg\tools\labelme_to_mmseg.py --input "%lbdir%" --img-out "%outimg%" --mask-out "%outmask%"
echo Done! You can retrain now
pause
goto menu

:full
echo.
echo === Full Workflow ===
echo.
echo Step 1/2: Model Prediction...
"%PYTHON%" mmseg\tools\predict_mmseg.py --config mmseg\configs\wheat\wheat_unet_r50.py --checkpoint mmseg\work_dirs\wheat_unet\iter_40000.pth --image to_predict --output predictions
echo.
echo Step 2/2: Convert to labelme...
"%PYTHON%" mmseg\tools\pred_to_labelme.py --pred predictions --image to_predict --output data_set\labelme_annotations
echo.
echo Done! Open data_set\labelme_annotations to edit JSON files in labelme
echo After editing, run option 3 to export as training data
pause
goto menu

:end
echo Goodbye!
pause

@echo off
:: AGRU 择时策略启动脚本
:: 解决 conda activate 导致的 libiomp5md.dll 冲突
:: 用法: run_env.bat run_daily_predict.py --date 2026-09-01

set KMP_DUPLICATE_LIB_OK=TRUE

:: 清理 PATH 中 base 环境的 Library\bin (冲突 DLL 来源)
set "CLEAN_PATH=%PATH%"
set "CLEAN_PATH=%CLEAN_PATH:D:\Miniforge\Library\bin;=%"
set "CLEAN_PATH=%CLEAN_PATH:D:\Miniforge\Library\mingw-w64\bin;=%"

:: aiquant 环境路径优先
set "PATH=D:\Miniforge\envs\aiquant;D:\Miniforge\envs\aiquant\Library\bin;D:\Miniforge\envs\aiquant\Scripts;%CLEAN_PATH%"

D:\Miniforge\envs\aiquant\python.exe %*

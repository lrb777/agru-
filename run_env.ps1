# AGRU 择时策略启动脚本
# 绕过 conda activate 导致的 DLL 冲突, 清理 PATH 中冲突的 base 库
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 移除 base 环境的 Library\bin (冲突 libiomp5md.dll 来源)
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -ne 'D:\Miniforge\Library\bin' -and $_ -ne 'D:\Miniforge\Library\mingw-w64\bin' }) -join ';'

# 确保 aiquant 环境路径在前
$aiquantPaths = @(
    'D:\Miniforge\envs\aiquant'
    'D:\Miniforge\envs\aiquant\Library\bin'
    'D:\Miniforge\envs\aiquant\Scripts'
)
$env:PATH = ($aiquantPaths -join ';') + ';' + $env:PATH

# 运行
& "D:\Miniforge\envs\aiquant\python.exe" @args

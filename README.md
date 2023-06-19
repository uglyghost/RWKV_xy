# RWKV_xy

RWKV_xy 是一个测试项目，主要涉及C++和CUDA的编程和应用。

## 配置环境

请按照以下步骤安装并设置您的环境：

### 工具安装

1. 安装VS2022构建工具，你可以在以下链接中下载：https://aka.ms/vs/17/release/vs_BuildTools.exe  选择桌面版C++。
2. 重新安装CUDA 11.7，记得安装VC++扩展。

### 执行脚本

使用"x64 native tools command prompt"（在开始菜单中找到）运行`train.py` or `main.py`。

### 环境变量设置

在Windows 11操作系统中，请按照以下步骤设置环境变量：

1. LD_LIBRARY_PATH 设置为 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64`
2. PATH 设置为 `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64`

## C++ 代码

请仔细阅读`./c++/RWKV-CUDA-main`内的内容以更好地理解C++的应用。

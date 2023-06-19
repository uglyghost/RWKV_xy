# RWKV_xy

RWKV_xy is a test project, focusing primarily on RWKV.
## Environment Setup

Please follow these steps to install and setup your environment:

### Tools Installation

1. Install VS2022 build tools. They can be downloaded from the following link: https://aka.ms/vs/17/release/vs_BuildTools.exe. Select Desktop C++.
2. Reinstall CUDA 11.7 and make sure to install the VC++ extensions.

### Script Execution

Run `main.py` or `train.py` using the "x64 native tools command prompt", which can be found in your start menu.

### Setting Environment Variables

On Windows 11, please set your environment variables as follows:

- Set `LD_LIBRARY_PATH` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64`
- Add `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64` to `PATH`

## C++ Code

Please read through `./c++/RWKV-CUDA-main` carefully to better understand the application of C++. 

Please feel free to contact us if you encounter any issues or have any suggestions.

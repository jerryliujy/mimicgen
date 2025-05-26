# !/bin/bash

set -e  # 遇到错误即退出

echo "🚀 Starting environment setup..."

# 1. 激活环境
echo "📦 Activating environment..."
# source activate
# conda deactivate
# conda activate dp

# 设置清华镜像源
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_HOST="pypi.tuna.tsinghua.edu.cn"

# 2. 安装常用科学计算包
echo "🔬 Installing scientific packages..."
pip install -i $PIP_INDEX --trusted-host $PIP_HOST \
    numba==0.56.4 \
    opencv-python==4.6.0.66 \
    cffi==1.15.1 \
    ipykernel==6.16 \
    zarr==2.12.0 \
    numcodecs==0.10.2 \
    h5py==3.7.0 \
    einops==0.4.1 \
    tqdm==4.64.1 \
    dill==0.3.5.1

# 3. 安装图像处理和媒体包
echo "🖼️ Installing image and media packages..."
pip install -i $PIP_INDEX --trusted-host $PIP_HOST \
    scikit-image==0.19.3 \
    imageio==2.22.0 \
    imageio-ffmpeg==0.4.7 \
    av==10.0.0

# 4. 安装ML框架相关
echo "🤖 Installing ML packages..."
pip install -i $PIP_INDEX --trusted-host $PIP_HOST \
    scikit-video==1.1.11 \
    hydra-core==1.2.0 \
    wandb==0.13.3 \
    diffusers==0.11.1 \
    huggingface_hub==0.10.1 \

# 5. 安装系统工具包
echo "🔧 Installing system packages..."
pip install -i $PIP_INDEX --trusted-host $PIP_HOST \
    termcolor==2.0.1 \
    psutil==5.9.2 \
    click==8.0.4 \
    cython==0.29.32 \
    cmake==3.24.3 

echo "✅ Pip packages installed successfully!"

conda install gym=0.21.0 \
    llvm-openmp=14


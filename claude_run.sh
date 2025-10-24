# 淘宝站
# export ANTHROPIC_AUTH_TOKEN=sk-mj6vDSGVxPri0a6NYjbpu949xYFWOTPMNFbykuzhfpctqH9a
# export ANTHROPIC_BASE_URL=https://core.claudemax.xyz
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda-13.0   # ⚠️ 强烈建议与 torch 的 cu128 匹配
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"   # 按你的 GPU 算力改
export USE_NINJA=1
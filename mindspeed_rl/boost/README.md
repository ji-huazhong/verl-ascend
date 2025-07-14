# Boost

Boost 是 MindSpeed RL 框架的核心模块，专为开源强化学习库提供昇腾 NPU 硬件加速能力。通过深度适配昇腾软件栈，Boost 显著提升了主流 RL 库（如 Verl、AReal）在 NPU 上的训练与推理性能，并扩展了其原生功能特性。

## 🚀 安装指南
### 环境要求
- 昇腾 Atlas 系列硬件（xxx)
- CANN 工具链xxx
- 对应 RL 框架（verl>=xxx)

### 快速安装
```bash
git clone https://github.com/mindspeed-ai/mindspeed_rl.git
cd mindspeed_rl/boost
pip install -r requirements.txt
python setup.py install
```

### 验证安装
```bash
python -c "import mindspeed_rl.boost; print(mindspeed_rl.boost.version())"
```

## 📖 使用示例
### Verl 框架加速
```bash
# 确保安装 verl
git clone https://github.com/mindspeed-ai/mindspeed_rl.git
cd mindspeed_rl/boost
bash verl_replace.sh
```
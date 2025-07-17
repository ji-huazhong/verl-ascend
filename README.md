- https://github.com/jiaqiw09/verl-dapo

# 使用方式

## 1、安装vllm和vllm-ascned
```bash
# vllm
git clone -b v0.9.1 https://github.com/vllm-project/vllm.git
pip install -r vllm/requirements/build.txt -i https://mirrors.aliyun.com/pypi/simple/#将里面的torch==2.7删除
pip install -r vllm/requirements/common.txt -i https://mirrors.aliyun.com/pypi/simple/
cd vllm
VLLM_TARGET_DEVICE=empty python setup.py develop
# vllm-ascend
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git + git checkout 02640b2f4b137cc75d1b6888697313676542cb00
cd vllm-ascend
pip install -v -e .
```

## 2、安装verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 218298720fadfc020c2fd37c7109025e48e29511
pip install -e .
```

## 3、安装插件
```bash
git clone https://github.com/ji-huazhong/mindspeed-rl.git
cd mindspeed-rl
pip install -e .
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```
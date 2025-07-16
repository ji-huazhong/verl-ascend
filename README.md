- https://github.com/jiaqiw09/verl-dapo

# 使用方式
## 1、安装verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 218298720fadfc020c2fd37c7109025e48e29511
pip install -e .
```

## 2、安装插件
```bash
git clone https://github.com/ji-huazhong/mindspeed-rl.git
cd mindspeed-rl
pip install -e .
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```
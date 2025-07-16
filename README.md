- https://github.com/jiaqiw09/verl-dapo

# 使用方式
## 1、安装verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 5cbad83792b53a7afc728dde739327721db95828
pip install -e .
```

## 2、安装插件
```bash
git clone https://github.com/ji-huazhong/mindspeed-rl.git
cd mindspeed-rl
python setup.py install
```

**注意**：安装插件前需要保证verl源码安装正确，否则插件不能生效。如果无法源码安装verl，可以手动将下述代码加入到verl_path/verl/__init__.py文件中：

```python
# NPU acceleration support added by mindspeed-rl plugin
from verl.utils.device import is_npu_available

if is_npu_available:
    from mindspeed_rl.boost import verl
    print("NPU acceleration enabled for verl")
```
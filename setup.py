import os
import sys
import sysconfig
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

# 插件注入逻辑
def inject_verl_plugin(custom_path=None):
    """将NPU加速支持注入到verl包中"""
    print("Starting verl plugin injection...")
    
    # 优先级：环境变量 > 自定义路径 > 自动查找
    if 'VERL_PATH' in os.environ:
        verl_path = os.path.join(os.environ['VERL_PATH'], "verl")
        print(f"Using verl path from environment variable: {verl_path}")
    elif custom_path:
        verl_path = custom_path
        print(f"Using custom verl path: {verl_path}")
    else:
        print("Searching for verl package automatically...")
        # 尝试多种方式查找verl安装路径
        paths_to_try = [
            sysconfig.get_paths()["purelib"],
            sysconfig.get_paths()["platlib"],
        ] + sys.path  # 搜索所有Python路径
        
        verl_path = None
        for path in paths_to_try:
            if not path:  # 跳过空路径
                continue
                
            candidate = os.path.join(path, "verl")
            if os.path.exists(candidate) and os.path.isdir(candidate):
                verl_path = candidate
                break
        
        # 使用pip show作为备用方案
        if not verl_path:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", "verl"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.splitlines():
                    if line.startswith("Location:"):
                        verl_path = os.path.join(line.split(": ")[1], "verl")
                        break
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"pip show failed: {e}")
    
    if not verl_path:
        print("Error: verl package not found. Please specify with VERL_PATH environment variable.")
        return False
    
    print(f"Found verl at: {verl_path}")
    
    init_file = os.path.join(verl_path, "__init__.py")
    if not os.path.exists(init_file):
        print(f"Error: verl initialization file not found: {init_file}")
        return False
    
    # 检查是否已经注入过
    import_content = """
# NPU acceleration support added by mindspeed-rl plugin
from verl.utils.device import is_npu_available

if is_npu_available:
    from mindspeed_rl.boost import verl
    print("NPU acceleration enabled for verl")
"""
    
    # 读取当前内容
    try:
        with open(init_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {init_file}: {e}")
        return False
    
    if import_content in content:
        print(f"Info: {init_file} already contains NPU acceleration import")
        return True
    
    # 添加注入内容
    try:
        with open(init_file, "a") as f:
            f.write(import_content)
        print(f"Successfully modified {init_file} to add NPU acceleration support")
        return True
    except Exception as e:
        print(f"Error writing to {init_file}: {e}")
        return False

# 自定义安装命令
class CustomInstallCommand(install):
    """自定义安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after standard install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        inject_verl_plugin(custom_path)

# 自定义开发模式安装命令
class CustomDevelopCommand(develop):
    """自定义开发模式安装命令"""
    def run(self):
        super().run()
        print("Running verl injection after develop install...")
        # 尝试从环境变量获取路径
        custom_path = os.environ.get('VERL_PATH', None)
        inject_verl_plugin(custom_path)

# 主安装函数
def main():
    print("Setting up mindspeed_rl...")
    
    # 尝试从命令行参数获取 --verl-path
    custom_path = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--verl-path='):
            custom_path = arg.split('=', 1)[1]
            # 移除这个参数
            sys.argv.pop(i)
            break
        elif arg == '--verl-path' and i + 1 < len(sys.argv):
            custom_path = sys.argv[i+1]
            # 移除这两个参数
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
    
    setup(
        name="mindspeed_rl",
        version="0.0.1",
        author="MindSpeed RL team",
        license="Apache 2.0",
        description="verl Ascend backend plugin",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
        ],
        python_requires=">=3.9",
        # install_requires=[
        #     "verl>=1.0.0",
        # ],
        entry_points={
            'verl.plugins': [
                'npu_acceleration = mindspeed_rl.boost:enable_npu_acceleration',
            ],
        },
        cmdclass={
            'install': CustomInstallCommand,
            'develop': CustomDevelopCommand,
        },
    )
    
    # 如果通过 setup.py 直接运行且指定了路径，执行注入
    if custom_path:
        print("Running direct injection from command line argument...")
        inject_verl_plugin(custom_path)

if __name__ == '__main__':
    main()
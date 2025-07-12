import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.command.build import build


class PreInstallHook:
    """A hacky workaround to inject verl_ascend into verl."""
    
    def run(self):
        print("[INFO] This installation will inject code into verl's __init__.py to load verl_ascend.")
        print("This is a hacky workaround until official plugin support is available.")
        
        try:
            import verl
        except ImportError:
            print("verl is not available. Please install verl first before installing verl_ascend.")
            sys.exit(1)

        try:
            verl_init_path = verl.__file__
            if not verl_init_path.endswith('__init__.py'):
                verl_init_path = os.path.join(os.path.dirname(verl_init_path), '__init__.py')
            
            if not os.path.exists(verl_init_path):
                raise FileNotFoundError(f"verl __init__.py file not found: {verl_init_path}")
        except Exception as e:
            print(f"Failed to locate verl installation: {str(e)}")
            sys.exit(1)

        try:
            if not os.access(verl_init_path, os.W_OK):
                raise PermissionError(f"Write permission denied: {verl_init_path}")
        except PermissionError as e:
            print("Insufficient permissions to modify verl!")
            print(f"  Details: {str(e)}")
            print("  Suggested solutions:")
            print("  1. Use administrator privileges (e.g., sudo pip install .)")
            print("  2. Check file permissions for the Verl installation directory")
        except Exception as e:
            print(f" Error checking permissions: {str(e)}")
    
        plugin_import_code = "import verl_ascend"
        try:
            with open(verl_init_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if plugin_import_code in content:
                print(f"verl_ascend import already exists in {verl_init_path}, skipping injection.")
                return
            
            # 写入导入代码
            with open(verl_init_path, 'a', encoding='utf-8') as f:
                import_code = """
# The following code is automatically added by the verl_ascend plugin
from verl.utils.device import is_npu_available
if is_npu_available():
    import verl_ascend
"""
                f.write(import_code)
            print(f"Successfully injected verl_ascend import into {verl_init_path}")
        except Exception as e:
            print(f" Failed to modify verl's __init__.py: {str(e)}")
            sys.exit(1)


class CustomInstall(install):
    def run(self):
        PreInstallHook().run()
        install.run(self)


class CustomBuild(build):
    def run(self):
        PreInstallHook().run()
        build.run(self)


setup(
    name="verl_ascend",
    verion="0.0.1",
    author="verl-Ascend team",
    license="Apache 2.0",
    description="verl Ascend backend plugin",
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
)

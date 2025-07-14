import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class PostInstallCommand(install):
    """A hacky workaround to inject the mindspeed_rl plugin into the verl."""

    user_options = install.user_options + [
        ("verl-path", None, "Specify the installation path of verl"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.verl_path = None
    
    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # 1. Execute default install process
        install.run(self)

        # 2. Determine verl lib path (prioritize user parameter, fallback to auto-detection)
        if self.verl_path:
            # 2.1. Use user-specified path
            if not os.path.isdir(self.verl_path):
                print(f"Error: Specified verl path does not exists: {self.verl_path}")
                sys.exit(1)
            print(f"Using user-specified verl path: {self.verl_path}")
        else:
            # 2.2. Auto-detect verl path
            try:
                import verl
                self.verl_path = verl.__path__[0]
                print(f"Automatically detected verl path: {self.verl_path}")
            except ImportError:
                print("Error: Failed to import verl. Please specify path using --verl-path")

        # 3. Modify verl initialization file
        self._modify_verl_init_file()

    def _modify_verl_init_file(self):
        init_file = os.path.join(self.verl_path, "__init__.py")

        # Check if file exists
        if not os.path.exists(init_file):
            print(f"Warning: verl initialization file not found: {init_file}")
            return
    
        # Define content to append
        import_content = """
# NPU acceleration support added by mindspeed-rl plugin
from verl.utils.device import is_npu_available

if is_npu_available:
    from mindspeed_rl.boost import verl
    print("NPU acceleration enabled for verl")
"""
        with open(init_file, "r") as f:
            if import_content in f.read():
                print(f"Info: {init_file} already contains npu acceleration import, skipping modification")
                return
        
        with open(init_file, "a") as f:
            f.write(import_content)
        print(f"Successfully modified {init_file} to add npu acceleration support")



setup(
    name="mindspeed_rl",
    verion="0.0.1",
    author="MindSpeed RL team",
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
    cmdclass={"install": PostInstallCommand},
)

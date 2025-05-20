from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # We expect CMake to be run manually for C++ components.
        # This class is a placeholder for Python packaging.
        # For a true C++ build via setup.py, this would need significant expansion.
        print("*********************************************************************************")
        print("SKIPPING C++ build through setup.py. Please build C++ components using CMake directly.")
        print(f"  cd {self.build_temp} && cmake {self.extensions[0].sourcedir} -DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.build_lib}/{self.extensions[0].name} -DPYTHON_EXECUTABLE={sys.executable} && make")
        print("*********************************************************************************")
        # super().run() # If you were to actually build

# Read requirements.txt for install_requires
requirements_path = this_directory / "requirements.txt"
install_requires = []
if requirements_path.is_file():
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
else:
    # Fallback or default dependencies if requirements.txt is not present
    install_requires = [
        'numpy>=1.21.0', # Updated numpy version
        'torch>=2.0.1,<2.2.0', # PyTorch with upper bound
        'tensorflow>=2.12.0,<2.14.0', # TensorFlow with upper bound
        'tvm==0.13.0', # Pinned TVM for now, as in CMake
        'structlog>=23.1.0,<24.0.0',
        'scipy>=1.9.0,<1.11.0',
        'matplotlib>=3.5.0,<3.8.0',
        'networkx>=2.8.0,<3.2.0',
        'pytest>=7.0.0,<7.5.0',
        'onnx>=1.13.0,<1.15.0', # Added ONNX
        'protobuf<4', # Often a dependency for ONNX/TF
    ]

setup(
    name="openoptimizer",
    version="0.1.0",
    author="OpenOptimizer Team",
    author_email="contact@openoptimizer.org", # Changed email
    description="A comprehensive neural network optimization framework",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/openoptimizer/openoptimizer",
    project_urls={
        'Documentation': 'https://openoptimizer.readthedocs.io/',
        'Source Code': 'https://github.com/openoptimizer/openoptimizer',
        'Bug Tracker': 'https://github.com/openoptimizer/openoptimizer/issues',
        'Discussions': 'https://github.com/openoptimizer/openoptimizer/discussions',
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    ext_modules=[CMakeExtension('openoptimizer._core', sourcedir='.')], # Assuming a _core module for C++ bindings
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires='>=3.11',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent', # More general
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12', # Adding 3.12 support
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Compilers',
    ],
    keywords="machine learning, neural network, optimization, compilation, compiler, deep learning, inference, TVM, MLIR, CUDA, edge computing",
    # Example: entry_points to create command-line tools
    entry_points={
        'console_scripts': [
            'openoptimizer-viz=openoptimizer.visualization.desktop.main:main', # Placeholder path
        ],
    },
)
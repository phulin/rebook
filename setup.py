from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()

files = ["inpaint", "newton", "collate", "feature_sign"]

extensions = [Extension(file_name, [f"./deps/{file_name}.pyx"]) for file_name in files]

modules = cythonize(extensions, language_level="3")
for e in modules:
    e.include_dirs.append(np.get_include())


setup(
    name="rebook",
    version="0.1.0",
    packages=["rebook"],
    package_dir={"rebook": "src/rebook"},
    author="Patrick Hulin",
    description="various book-scan-processing programs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phulin/rebook",
    ext_modules=modules,
    include_dirs=[np.get_include()],
    install_requires=[
        "certifi>=2020.4.5.1",
        "cycler>=0.10.0",
        "Cython>=0.29.19",
        "decorator>=4.4.2",
        "fpdf>=1.7.2",
        "freetype-py>=2.1.0.post1",
        "imagecodecs>=2020.2.18",
        "imageio>=2.8.0",
        "joblib>=0.15.1",
        "kiwisolver>=1.2.0",
        "matplotlib>=3.2.1",
        "networkx>=2.4",
        "numpy>=1.18.4",
        "opencv-python>=4.2.0.34",
        "Pillow>=7.1.2",
        "pyparsing>=2.4.7",
        "python-dateutil>=2.8.1",
        "PyWavelets>=1.1.1",
        "rawpy>=0.14.0",
        "scikit-image>=0.17.2",
        "scikit-learn>=0.23.1",
        "scipy>=1.4.1",
        "six>=1.15.0",
        "threadpoolctl>=2.0.0",
        "tifffile>=2020.5.11",
    ],
)

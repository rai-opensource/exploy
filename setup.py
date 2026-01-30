"""Setup file for the exporter package."""

from setuptools import find_packages, setup

setup(
    name="exporter",
    version="0.1.0",
    description="ONNX environment exporter",
    author="Robotics and AI Institute LLC dba RAI Institute",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "onnx",
        "onnxruntime",
        "numpy",
    ],
)

from setuptools import setup, find_packages

setup(
    name="gcdetr",
    version="0.1.0",
    description="GC-DETR: Geometry-Conditioned Real-Time Object Detection for UAV-based Open-Water Scenarios",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "Pillow>=8.0.0",
    ],
)

"""Setup script for YouAI package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="youai",
    version="0.2.0",
    author="YouAI Contributors",
    description="Train Your Own Language Model From Scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youai/youai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "tokenizers>=0.13.0",
    ],
    extras_require={
        "datasets": ["datasets>=2.12.0"],
        "wandb": ["wandb>=0.15.0"],
        "web": ["flask>=3.0.0", "flask-cors>=4.0.0"],
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.16.0"],
        "all": [
            "datasets>=2.12.0",
            "wandb>=0.15.0",
            "flask>=3.0.0",
            "flask-cors>=4.0.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "youai=youai.cli:main",
        ],
    },
)

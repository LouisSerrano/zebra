from setuptools import setup, find_packages
import os

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt if it exists
install_requires = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "torchvision",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "tqdm>=4.62.0",
    "hydra-core>=1.1.0",
    "omegaconf>=2.1.0",
    "wandb>=0.12.0",
    "lightning>=2.0.0",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "accelerate>=0.20.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "huggingface-hub>=0.15.0",
    "vector-quantize-pytorch>=1.11.0",
    "beartype"
]

# Development dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.0.0",
    ],
}

setup(
    name="zebra",
    version="0.1.0",
    author="Louis Serrano",
    author_email="louis.serrano@isir.upmc.fr",
    description="A library for training and using tokenizers with LLaMA models for physical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisSerrano/zebra",
    project_urls={
        "Bug Tracker": "https://github.com/LouisSerrano/zebra/issues",
        "Documentation": "https://zebra.readthedocs.io/",
        "Source Code": "https://github.com/LouisSerrano/zebra",
    },
    packages=find_packages(include=["zebra", "zebra.*"]),
    package_data={
        "zebra": [
            "configs/**/*.yaml",
            "configs/**/*.md",
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "zebra-train=zebra.cli.train:main",
            "zebra-test=zebra.cli.test:main",
        ],
    },
) 

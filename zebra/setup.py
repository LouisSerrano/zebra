from setuptools import setup, find_packages

setup(
    name="zebra",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pytorch-lightning>=2.0.0",
        "wandb>=0.15.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "einops>=0.6.0",
        "numpy>=1.24.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for training and using tokenizers with LLaMA models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zebra",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 
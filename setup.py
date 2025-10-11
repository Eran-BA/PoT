"""
Setup script for PoT: Pointer-over-Heads Transformer.

Installation:
    pip install -e .

Author: Eran Ben Artzy
License: Apache 2.0
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="pot-parser",
    version="0.1.0",
    author="Eran Ben Artzy",
    description="Pointer-over-Heads Transformer for dependency parsing with adaptive routing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eran-BA/PoT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)

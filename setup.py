"""Setup script for scp-bench."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f
                    if line.strip() and not line.startswith("#")]

setup(
    name="scp-bench",
    version="0.1.0",
    author="AIResearcher20",
    description="Reproducible ML benchmarking for single-cell proteomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AIResearcher20/scp-bench",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

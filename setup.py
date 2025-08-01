from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scrna-functions",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Single-cell RNA analysis functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scrna-functions",
    packages=find_packages(),
    package_data={
        'scrna_functions': ['*.json', '*.txt'],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "seaborn",
        "anndata",
        "scanpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intelligent-search-engine",
    version="1.0.0",
    author="Your Name",
    author_email="tylerelyt@gmail.com",
    description="A machine learning-powered intelligent search engine with CTR prediction and MLOps support",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tylerelyt/test_bed",
    project_urls={
        "Bug Tracker": "https://github.com/tylerelyt/test_bed/issues",
        "Documentation": "https://github.com/tylerelyt/test_bed/docs",
        "Source Code": "https://github.com/tylerelyt/test_bed",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "coverage>=6.0.0",
        ],
        "ml": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "sentence-transformers>=2.2.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.18.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "prometheus-client>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "intelligent-search=search_engine.cli:main",
            "search-engine=search_engine.start_system:main",
        ],
    },
    include_package_data=True,
    package_data={
        "search_engine": [
            "data/*.json",
            "models/*.pkl",
            "models/*.json",
            "static/*",
            "templates/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "search engine",
        "machine learning",
        "information retrieval",
        "CTR prediction",
        "MLOps",
        "natural language processing",
        "text mining",
        "ranking",
        "indexing",
    ],
) 
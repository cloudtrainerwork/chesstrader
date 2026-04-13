"""
Setup configuration for ChessTrader Options AI package.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "ChessTrader - AI-powered options trading system with chess-inspired neural architecture"

setup(
    name="chesstrader",
    version="1.0.0",
    description="AI-powered options trading system with chess-inspired neural architecture",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ChessTrader Development Team",
    author_email="contact@chesstrader.ai",
    url="https://github.com/chesstrader/chesstrader",
    project_urls={
        "Documentation": "https://chesstrader.readthedocs.io",
        "Source": "https://github.com/chesstrader/chesstrader",
        "Tracker": "https://github.com/chesstrader/chesstrader/issues",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chesstrader=src.cli.main:app",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "options trading",
        "artificial intelligence",
        "machine learning",
        "neural networks",
        "chess ai",
        "financial analysis",
        "backtesting",
        "algorithmic trading",
        "quantitative finance",
        "regime detection",
        "reinforcement learning",
    ],
    zip_safe=False,
    platforms=["any"],
    license="MIT",
)
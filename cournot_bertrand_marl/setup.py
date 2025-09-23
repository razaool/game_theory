from setuptools import setup, find_packages

setup(
    name="cournot-bertrand-marl",
    version="0.1.0",
    description="Multi-Agent Reinforcement Learning for Cournot and Bertrand Competition",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "sympy>=1.9.0",
        "pandas>=1.3.0",
        "torch>=1.12.0",
        "ray[rllib]>=2.0.0",
        "pettingzoo>=1.22.0",
        "gym>=0.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "nashpy>=0.0.21",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

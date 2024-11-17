from setuptools import setup, find_packages

setup(
    name="legoml",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A modular and flexible library for core machine learning tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/legoml",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

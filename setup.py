from setuptools import setup, find_packages

setup(
    name="legoml",
    version="0.1.1",
    author="Thomas Rauter",
    author_email="rauterthomas0@gmail.com",
    description="A modular and flexible library for core machine learning"
                " tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Thomas-Rauter/legoml",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.0.0",
        "ipython>=8.0.0"
    ],
    extras_require={
        "torch": ["torch>=2.4.0"],  # PyTorch-related extras
        "tensorflow": ["tensorflow>=2.13.0"],  # TensorFlow-related extras
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

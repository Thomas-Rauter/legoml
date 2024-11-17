from setuptools import setup, find_packages

setup(
    name="legoml",
    version="0.1.0",
    author="Thomas Rauter",
    author_email="rauterthomas0@gmail.com",
    description="A modular and flexible library for core machine learning"
                " tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Thomas-Rauter/legoml",
    packages=find_packages(),
    install_requires=[
        "filelock>=3.15.0",
        "fsspec>=2024.9.0",
        "Jinja2>=3.0.0",
        "MarkupSafe>=3.0.0",
        "mpmath>=1.2.0",
        "networkx>=3.3.0",
        "sympy>=1.12.0",
        "torch>=2.4.0",
        "typing_extensions>=4.11.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

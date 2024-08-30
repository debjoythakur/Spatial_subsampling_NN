#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name="spatial_subsampling",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "pandas",
        "tensorflow",
        "torch_geometric",
        "scikit-learn",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for spatial subsampling using neural networks.",
    url="https://github.com/yourusername/spatial_subsampling",  # Replace with your GitHub repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


from setuptools import setup, find_packages

setup(
    name="AugmentedDataLoader",
    version="3.0.4",  
    author="Ciro B. Raggio, P. Zaffino",
    author_email="ciro.raggio@kit.edu",
    description="Medical image augmentation tool integrated with PyTorch & MONAI.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ciroraggio/AugmentedDataLoader", 
    packages=find_packages(include=["AugmentedDataLoader", "AugmentedDataLoader.*"]),
    install_requires=[
        "monai",
        "torch",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
from setuptools import setup, find_packages


setup(
    name="open3d_wrapper",
    version="1.0",
    install_requires=["open3d", "numpy", "trimesh", "scipy"],
    packages=find_packages(),
)

from setuptools import setup, find_packages
setup(
    name="temporal-stategraph",
    version="1.0.0",
    description="Calibrated decay for stale-memory suppression in long-horizon AI agents",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24", "scipy>=1.10", "matplotlib>=3.7"],
)

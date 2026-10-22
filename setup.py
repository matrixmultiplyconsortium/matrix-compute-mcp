from setuptools import find_packages, setup

setup(
    name="matrix-compute-mcp",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["matrix_compute_mcp*"]),
    entry_points={
        "console_scripts": [
            "matrix-compute-mcp = matrix_compute_mcp.server:main",
        ],
    },
)

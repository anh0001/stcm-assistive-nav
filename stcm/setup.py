from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


package_name = "stcm"
base_dir = Path(__file__).parent


def read_requirements():
    req_file = base_dir / "requirements.txt"
    if not req_file.exists():
        return ["setuptools"]
    with req_file.open("r", encoding="utf-8") as handle:
        requirements = [
            line.strip() for line in handle.readlines() if line.strip() and not line.startswith("#")
        ]
    requirements.insert(0, "setuptools")
    return requirements


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["stcm*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=read_requirements(),
    zip_safe=False,
    maintainer="Anhar Risnumawan",
    maintainer_email="anhrisn@gmail.com",
    description="Semantic Topological Cognitive Mapping package for ROS 2 Humble.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_map_builder = stcm.nodes.semantic_map_builder:main",
            "semantic_map_updater = stcm.nodes.semantic_map_updater:main",
            "stcm_download_checkpoints = stcm.tools.checkpoint_manager:main",
        ],
    },
)

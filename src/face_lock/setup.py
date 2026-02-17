from setuptools import find_packages, setup
import os
from glob import glob

package_name = "face_lock"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FaceLock Developer",
    maintainer_email="user@example.com",
    description="FaceLock robot control package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "face_lock_manager = face_lock.face_lock_manager:main",
            "arm_controller = face_lock.arm_controller:main",
            "pi_hardware = face_lock.pi_hardware:main",
            "camera = face_lock.camera:main",
            "face_recognition = face_lock.face_recognition:main",
        ],
    },
)

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['opencv-python', 'numpy', 'google-cloud-storage==1.16.1']

setup(
    name='training',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)

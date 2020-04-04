import os
import os.path
from setuptools import find_packages
from setuptools import setup
def find_requires():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    requirements = []
    with open('{0}/requirements.txt'.format(dir_path), 'r') as reqs:
        requirements = reqs.readlines()
    return requirements

if __name__ == "__main__":
    setup(
        name="crosspredict",
        version="0.0.1",
        description='package for easy crossvalidation',
        packages=find_packages(),
        install_requires=find_requires(),
        include_package_data=True,
        # entry_points={
        #     'console_scripts': [
        #         'my_command = my_package.cli:main',
        #     ],
        # },
    )
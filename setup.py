import os
import os.path
from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

def find_requires():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    requirements = []
    with open('{0}/requirements.txt'.format(dir_path), 'r') as reqs:
        requirements = reqs.readlines()
    return requirements

if __name__ == "__main__":
    setup(
        name="crosspredict",
        version="0.1.0",
        author="Vladislav Boyadzhi",
        author_email="vladislav.boyadzhi@gmail.com",
        description='package for easy crossvalidation',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=find_requires(),
        include_package_data=True,
        #download_url='https://github.com/crosspredict/crosspredict/archive/0.0.1.tar.gz',
        # entry_points={
        #     'console_scripts': [
        #         'my_command = my_package.cli:main',
        #     ],
        # },
    )
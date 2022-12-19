import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit('Sorry, Python < 3.8 is not supported.')

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="mlflow_er",
    packages=[package for package in find_packages()],
    version="0.0.1",
    license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
    description="teeny-tiny mlflow helper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ricardo de Azambuja",
    author_email="ricardo.azambuja@gmail.com",
    url="https://github.com/ricardodeazambuja/mlflow-er",
    download_url="https://github.com/ricardodeazambuja/mlflow-er/archive/refs/tags/v0.0.1.tar.gz",
    keywords=['mlflow', 'logging', 'ML', 'experiments'],
    install_requires=['mlflow>=2.0'],
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Topic :: Education',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
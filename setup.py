import codecs
import os
import re

from setuptools import setup, find_packages

NAME = "hydro-tune"
URL = "https://github.com/S-Lab-System-Group/Hydro"
AUTHOR = "Qinghao Hu"
EMAIL = "qinghao.hu@ntu.edu.sg"
PYTHON_REQUIRES = ">=3.8"
KEYWORDS = "AutoML, HPO"
INSTALL_REQUIRES = ["ray[air]==2.3.1", "torch>=2.0", "transformers>=4.30", "grpcio==1.51.1", "py3nvml"]
CLASSIFIERS = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file. Assume UTF-8 encoding.
    """
    cwd = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(cwd, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(os.path.join("hydro", "__init__.py"))


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META_FILE, re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        version=find_meta("version"),
        packages=find_packages(),
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        # package_data=PACKAGE_DATA,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
    )

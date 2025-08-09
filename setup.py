from setuptools import setup, find_packages
import pathlib

# Project metadata
NAME = "axiomelectrus"
VERSION = "1.1.2"
DESCRIPTION = "Electrus is a lightweight, MongoDB-style asynchronous and synchronous database module designed for Python."
AUTHOR = "Pawan Kumar"
AUTHOR_EMAIL = "aegis.invincible@gmail.com"
URL = "https://github.com/axiomchronicles/electrus"
LICENSE = "MIT"
REQUIRES_PYTHON = ">=3.6"
KEYWORDS = [
    "database", "nosql", "json", "mongodb", "async", "synchronous", "lightweight",
    "electrus", "python-database", "data-storage", "indexing", "file-based-db"
]
REQUIRED = [
    "aiofiles>=0.6.0",
    "numpy>=1.20.0",
    "filelock>=3.0.0"
]

# Read README.md for long_description
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    packages=find_packages(include=["axiomelectrus", "axiomelectrus.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=KEYWORDS,
    project_urls={
        "Documentation": "https://github.com/axiomchronicles/electrus/wiki",
        "Source": "https://github.com/axiomchronicles/electrus",
        "Tracker": "https://github.com/axiomchronicles/electrus/issues",
        "Changelog": "https://github.com/axiomchronicles/electrus/releases",
    },
)
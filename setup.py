from setuptools import setup, find_packages
from deepeval._version import __version__

setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/mr-gpt/evals",
    author="Twilix",
    author_email="jacky@twilix.io",
    description="Eval",
    packages=find_packages(),
    # TODO - make pandas an 'extra' requirement in the future
    install_requires=[
        "requests",
        "tqdm",
        "transformers",
        "pytest",
        "tabulate",
    ],
)

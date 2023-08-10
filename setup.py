from evals import __version__
from setuptools import setup, find_packages

setup(
    name="Evals",
    version=__version__,
    url="https://github.com/mr-gpt/evals",
    author="Twilix",
    author_email="jacky@twilix.io",
    description="Eval",
    packages=find_packages(),
    install_requires=["requests", "tqdm"],
)

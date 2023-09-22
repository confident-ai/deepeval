from setuptools import find_packages, setup

from deepeval._version import __version__

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/mr-gpt/deepeval",
    author="Confident AI",
    author_email="jacky@twilix.io",
    description="Deep eval provides evaluation platform to accelerate development of LLMs and Agents",
    packages=find_packages(),
    # TODO - make pandas an 'extra' requirement in the future
    install_requires=[
        "requests",
        "tqdm",
        "transformers",
        "pytest",
        "tabulate",
        "sentence-transformers",  # for similarity
        "pytest",
        "typer",
        "rich",
        "protobuf<=3.20.5",
        "pandas",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "bias": [
            "tensorflow",  # for bias
            "Dbias",  # for bias
        ],
        "toxic": [
            "detoxify",  # for toxic classifier
        ],
        "dev": ["black"],
    },
    entry_points={
        "console_scripts": [
            "deepeval = deepeval.cli.main:app",
        ],
        "pytest11": [
            "plugins = deepeval.plugins.plugin",
        ],
    },
)

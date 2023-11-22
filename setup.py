from setuptools import find_packages, setup

from deepeval._version import __version__

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()


setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/confident-ai.com/deepeval",
    author="Confident AI",
    author_email="jeffreyip@confident-ai.com",
    description="The open-source evaluation framework for LLMs.",
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
        "typer==0.9.0",
        "rich",
        "protobuf<=3.20.5",
        "pandas",
        "pydantic",  # loosen pydantic requirements as we support multiple
        "sentry-sdk",
        "pytest-xdist",
        "portalocker",
        "langchain",
        "rouge_score==0.1.2",
        "nltk==3.8.1",
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

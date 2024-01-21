from setuptools import find_packages, setup

from deepeval._version import __version__

setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/confident-ai.com/deepeval",
    author="Confident AI",
    author_email="jeffreyip@confident-ai.com",
    description="The open-source evaluation framework for LLMs.",
    packages=find_packages(),
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
        "protobuf==4.25.1",
        "pydantic",  # loosen pydantic requirements as we support multiple
        "sentry-sdk",
        "pytest-xdist",
        "portalocker",
        "langchain",
        "langchain-core",
        "langchain_openai",
        "rouge_score==0.1.2",
        "nltk==3.8.1",
        "ragas",
    ],
    extras_require={
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

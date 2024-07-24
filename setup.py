from setuptools import find_packages, setup

from deepeval._version import __version__

setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/confident-ai/deepeval",
    author="Confident AI",
    author_email="jeffreyip@confident-ai.com",
    description="The open-source evaluation framework for LLMs.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        "tqdm",
        "pytest",
        "tabulate",
        "typer",
        "rich",
        "protobuf==4.25.1",
        "pydantic",  # loosen pydantic requirements as we support multiple
        "sentry-sdk",
        "pytest-repeat",
        "pytest-xdist",
        "portalocker",
        "langchain",
        "langchain-core",
        "langchain_openai",
        "ragas",
        "docx2txt~=0.8",
        "importlib-metadata>=6.0.2",
        "tenacity~=8.4.1",
        "opentelemetry-api==1.24.0",
        "opentelemetry-sdk==1.24.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.24.0",
        "grpcio==1.63.0",
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

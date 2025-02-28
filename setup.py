from setuptools import find_packages, setup
from pathlib import Path
from deepeval._version import __version__

# Read the README file
long_description = (Path(__file__).parent / "README.md").read_text(
    encoding="utf-8"
)

setup(
    name="deepeval",
    version=__version__,
    url="https://github.com/confident-ai/deepeval",
    author="Confident AI",
    author_email="jeffreyip@confident-ai.com",
    description="The Open-Source LLM Evaluation Framework.",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9, <3.14",
    install_requires=[
        "requests",
        "tqdm",
        "pytest",
        "tabulate",
        "typer",
        "rich",
        "protobuf",
        "pydantic",  # loosen pydantic requirements as we support multiple
        "sentry-sdk",
        "pytest-repeat",
        "pytest-xdist",
        "portalocker",
        "langchain",
        "llama-index",
        "langchain-core",
        "langchain_openai",
        "langchain-community",
        "docx2txt~=0.8",
        "importlib-metadata>=6.0.2",
        "tenacity<=9.0.0",
        "opentelemetry-api>=1.24.0,<2.0.0",
        "opentelemetry-sdk>=1.24.0,<2.0.0",
        "opentelemetry-exporter-otlp-proto-grpc>=1.24.0,<2.0.0",
        "grpcio>=1.67.1,<2.0.0",
        "nest-asyncio",
        "datasets",
        "ollama",
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

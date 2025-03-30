.PHONY: install

# Variables
PROJECT_NAME := deepeval

install:
	poetry install
	poetry shell
	pre-commit install

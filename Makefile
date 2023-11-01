.PHONY: format


ENV_PATH := ./configs/environment/
ENV_NAME := waste-semantic-segmentation
PYTHON_VERSION := 3.10


# Package manager targets
create_env: 
	pyenv install $(PYTHON_VERSION) --skip-existing
	pyenv local $$(pyenv latest -k $(PYTHON_VERSION))
	poetry config virtualenvs.in-project true
	poetry config virtualenvs.prefer-active-python true
	poetry install

prod_env:
	poetry install --only main

dev_env:
	poetry install


# Linting targets
lint_check:
	poetry run ruff check .

lint_fix: 
	poetry run ruff check --fix .

format: 
	poetry run ruff format .

lint_and_format: lint_fix format


# Pre-commit targets
initialize_precommit:
	poetry run pre-commit install --config configs/pre-commit/.pre-commit-config.yaml
	
autoupdate_precommit:
	poetry run pre-commit autoupdate --config configs/pre-commit/.pre-commit-config.yaml
	make initialize_precommit


# Version control targets
address_conflicts:
	git fetch
	git merge
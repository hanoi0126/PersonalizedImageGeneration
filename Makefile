.PHONY: lint
lint:
	poetry run ruff format fastcomposer
	poetry run ruff check --fix fastcomposer
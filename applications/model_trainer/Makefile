TEST-WORKERS=2

.PHONY: black
black:
	poetry run black --check trainer tests

.PHONY: black-lint
black-lint:
	poetry run black trainer tests

.PHONY: flake8
flake8:
	poetry run flake8 trainer tests

.PHONY: isort
isort:
	poetry run isort --check-only trainer tests

.PHONY: isort-lint
isort-lint:
	poetry run isort trainer tests

.PHONY: mypy
mypy:
	poetry run mypy trainer

.PHONY: test
test:
	poetry run pytest tests --cov=trainer --cov-report term-missing --durations 5

.PHONY: lint
lint:
	$(MAKE) black-lint
	$(MAKE) isort-lint

.PHONY: test-all
test-all:
	$(MAKE) black
	$(MAKE) flake8
	$(MAKE) isort
	$(MAKE) mypy
	$(MAKE) test
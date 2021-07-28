TEST-WORKERS=2

.PHONY: black
black:
	poetry run black --check imap tests

.PHONY: black-lint
black-lint:
	poetry run black imap tests

.PHONY: flake8
flake8:
	poetry run flake8 imap tests

.PHONY: isort
isort:
	poetry run isort --check-only imap tests

.PHONY: isort-lint
isort-lint:
	poetry run isort imap tests

.PHONY: mypy
mypy:
	poetry run mypy imap

.PHONY: test
test:
	poetry run pytest tests --cov=imap --cov-report term-missing --durations 5

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
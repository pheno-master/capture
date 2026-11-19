.venv:
	@python3 -m venv .venv

.venv/installed: .venv
	@.venv/bin/python3 -m pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
	touch .venv/installed

test: .venv/installed
	@.venv/bin/python3 -m pytest test

run: .venv/installed
	@.venv/bin/python3 -m main

.PHONY: test run

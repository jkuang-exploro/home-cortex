.PHONY: test

test:
	.venv/bin/python -m pytest -q
	cd src/home_media && .venv/bin/python -m pytest -q

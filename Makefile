SHELL := /bin/bash
include server/.env

start:
	cd server && docker-compose up

start-gpu:
	cd server && docker-compose -f docker-compose.yml -f gpu.yml up

build:
	cd server && docker-compose build

build-gpu:
	cd server && docker-compose -f docker-compose.yml -f gpu.yml build

format:
	autoflake -i **/*.py
	isort -i **/*.py
	yapf -i **/*.py

test:
	cd common/schemas/schemas && python3 -m pytest

flower:
	celery flower --broker=redis://127.0.0.1:63791 --address=127.0.0.1 

docs:
	sphinx-autobuild doc/source doc/build

docs_release:
	sphinx-build -aEW doc/source doc/build
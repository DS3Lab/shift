SHELL := /bin/bash
include .myenv
start:
	docker-compose up
start-gpu:
	docker-compose -f docker-compose.yml -f gpu.yml up
build:
	docker-compose build
build-gpu:
	docker-compose -f docker-compose.yml -f gpu.yml build
format:
	autoflake -i **/*.py
	isort -i **/*.py
	yapf -i **/*.py
test:
	cd common/schemas/schemas && python3 -m pytest
flower:
	celery flower --broker=redis://127.0.0.1:63791 --address=127.0.0.1 
	
set-dev-envs:
	set -o allexport
	source .env
	source .myenv
	source py-dev.env
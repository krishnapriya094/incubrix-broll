.PHONY: install test run compose-up compose-test batch

install:
	pip install -r requirements-dev.txt

test:
	./run_tests.sh

run:
	uvicorn app.main:app --reload --port 8000

compose-up:
	docker compose up --build

compose-test:
	docker compose run --rm api python -m pytest -q

batch:
	python -m app.batch_runner --input ./testdata --out ./batch_results

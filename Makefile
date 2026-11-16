PY ?= python3
export PYTHONPATH := src

.PHONY: install demo generate backtest train report forecast-example test lint serve docker clean

install:
	$(PY) -m pip install -r requirements-dev.txt

## Regenerate the SYNTHETIC data, backtest every model, publish a registry version and redraw the README charts
demo: clean generate backtest train report forecast-example

generate backtest train report:
	$(PY) -m salesforecast $@

forecast-example:
	$(PY) -m salesforecast forecast --store S03 --category snacks --horizon 7
	$(PY) -m salesforecast forecast --store S03 --horizon 7

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	$(PY) -m salesforecast serve

docker:
	docker compose up --build

clean:
	rm -rf data models reports

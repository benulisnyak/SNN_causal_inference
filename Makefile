.PHONY: install install-full test lint check-scripts demo figures

install:
	python -m pip install -e ".[dev]"

install-full:
	python -m pip install -e ".[training,dev]"

test:
	pytest

lint:
	ruff check src tests

check-scripts:
	python -m compileall -q experiments analysis

demo:
	python -m snn_connectivity inspect-network networks/network_N100_p24_CC01_1.yaml
	python -m snn_connectivity inspect-spikes examples/synthetic_fdata_N100_demo.txt --expected-n 100

figures:
	python -m snn_connectivity make-example-figures \
		--network networks/network_N100_p24_CC01_1.yaml \
		--spikes examples/synthetic_fdata_N100_demo.txt \
		--output-dir results/figures

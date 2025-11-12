.PHONY: setup small-run build-dataset train evaluate explain error-analysis clean

setup:
	# Create/refresh the Poetry virtual environment and install deps.
	poetry install

small-run:
	# End-to-end run using the starter config on modest hardware (8–16 GB RAM).
	poetry run python -m irp_phishxai.cli.build_dataset --config experiments/configs/starter.yaml
	poetry run python -m irp_phishxai.cli.train --config experiments/configs/starter.yaml --which all
	poetry run python -m irp_phishxai.cli.evaluate --config experiments/configs/starter.yaml
	poetry run python -m irp_phishxai.cli.explain --config experiments/configs/starter.yaml

build-dataset:
	poetry run python -m irp_phishxai.cli.build_dataset --config experiments/configs/starter.yaml

train:
	poetry run python -m irp_phishxai.cli.train --config experiments/configs/starter.yaml --which all
	#poetry run python -m irp_phishxai.cli.train --config experiments/configs/starter.yaml --which all --models rf xgb lgbm lr dt

evaluate:
	poetry run python -m irp_phishxai.cli.evaluate --config experiments/configs/starter.yaml

explain:
	poetry run python -m irp_phishxai.cli.explain --config experiments/configs/starter.yaml

error-analysis:
	poetry run python -m irp_phishxai.cli.error_analysis --config experiments/configs/starter.yaml

clean:
	# Remove generated artifacts to reset the project.
	rm -rf data/interim/* models/* experiments/runs/* reports/figures/* reports/tables/*
	#rm -rf data/interim/* data/processed/* models/* experiments/runs/* reports/figures/* reports/tables/*

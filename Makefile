UNIVERSE ?= combined
CONFIG   ?= ltc_config.toml
CHECKPOINT ?= checkpoints/best_val_sharpe.pt

.PHONY: help install train test lint format infer eval push-hf clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install dev + test dependencies
	pip install -r requirements-dev.txt

train:  ## Train the LTC model  (UNIVERSE=fi|equity|combined)
	python train.py --universe $(UNIVERSE) --config $(CONFIG)

eval:  ## Evaluate on test set
	python evaluate.py --universe $(UNIVERSE) --checkpoint $(CHECKPOINT)

infer:  ## Run daily inference and publish to HF
	python infer_daily.py --universe $(UNIVERSE)

test:  ## Run all unit + integration tests
	pytest --cov=. --cov-report=term-missing -q

lint:  ## Lint with ruff
	ruff check .

format:  ## Auto-format with black
	black .

push-hf:  ## Push results to Hugging Face dataset
	python publisher.py --universe $(UNIVERSE)

clean:  ## Remove pycache and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache

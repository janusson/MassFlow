.DEFAULT_GOAL := help
.PHONY: help lock-check lint format-check format typecheck test test-core test-scientific test-optional test-cov build clean all

# ── Tooling ──────────────────────────────────────────────────────────────────
UV := uv run

# ── Default ──────────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Lockfile, Lint & Format ──────────────────────────────────────────────────
lock-check: ## Verify uv.lock matches pyproject.toml
	uv lock --check

lint: ## Run ruff linter
	$(UV) ruff check .

format-check: ## Check formatting (dry-run)
	$(UV) ruff format --check .

format: ## Auto-format code with ruff
	$(UV) ruff format .

# ── Type checking ────────────────────────────────────────────────────────────
typecheck: ## Run mypy static type checker
	$(UV) mypy .

# ── Tests ────────────────────────────────────────────────────────────────────
test: ## Run the default release test suite
	$(UV) pytest

test-core: ## Run v0.1 stable contract tests (-m core)
	$(UV) pytest -m core -v

test-scientific: ## Run scientific-validation test group (-m scientific)
	$(UV) pytest -m scientific -v

test-optional: ## Run optional-dependency test group (-m optional)
	$(UV) pytest -m optional -v

test-cov: ## Run release test suite with coverage report (threshold 80%)
	$(UV) pytest \
		--cov=src/MassFlow \
		--cov-report=term \
		--cov-report=html \
		--cov-report=xml \
		--cov-fail-under=80 \
		-v

# ── Build & Clean ────────────────────────────────────────────────────────────
build: ## Build wheel and sdist via uv
	uv build

clean: ## Remove build artifacts, caches, and coverage output
	@echo "Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info .eggs
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

# ── CI Pipeline ──────────────────────────────────────────────────────────────
all: lock-check lint format-check typecheck test-cov test-scientific ## Run the full CI verification pipeline

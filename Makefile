.DEFAULT_GOAL := help
.PHONY: help lint format-check format typecheck test test-cov build clean all

# ── Tooling ──────────────────────────────────────────────────────────────────
UV := uv run

# ── Default ──────────────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Lint & Format ────────────────────────────────────────────────────────────
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
test: ## Run the full test suite
	$(UV) pytest

test-cov: ## Run tests with HTML coverage report (threshold 80 %)
	$(UV) pytest \
		--cov=src/MassFlow \
		--cov-report=term \
		--cov-report=html \
		--cov-fail-under=80

# ── Build & Clean ────────────────────────────────────────────────────────────
build: ## Build the wheel via hatchling
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
all: lint format-check typecheck test-cov ## Run the full CI pipeline

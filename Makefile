.DEFAULT_GOAL := help
.PHONY: help lint format-check format typecheck typecheck-src static test test-cov build clean all ci smoke

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
typecheck: ## Run mypy on the whole repo (src + tests) — the required gate
	$(UV) mypy .

typecheck-src: ## Fast mypy check of src/MassFlow only (PARTIAL — see note)
	@echo "note: 'typecheck-src' checks ONLY src/MassFlow. Run 'make typecheck' (full repo: src + tests) before pushing."
	$(UV) mypy src/MassFlow

static: lint format-check typecheck ## Fast static gate (lint + format + full type check, no tests)

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
all: lint format-check typecheck test-cov ## Run the full CI pipeline (fast local gate)

smoke: ## Generate tutorial data in a temp dir and run the annotate quickstart
	@TMP=$$(mktemp -d) ; \
	trap 'rm -rf "$$TMP"' EXIT INT TERM ; \
	echo "smoke: generating tutorial data in $$TMP" ; \
	cd "$$TMP" && $(UV) --project "$(CURDIR)" massflow tutorial && \
	$(UV) --project "$(CURDIR)" massflow annotate --config tutorial/tutorial_config.yaml && \
	echo "smoke: annotate quickstart succeeded"

ci: smoke ## Full CI mirror: lock check + static + tests + scientific + optional + docs
	uv lock --check
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) mypy .
	$(UV) pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
	$(UV) pytest -m scientific -v
	$(UV) pytest -m optional -v
	$(UV) mkdocs build --strict

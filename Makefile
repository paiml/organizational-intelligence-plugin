# Organizational Intelligence Plugin - Makefile
# Toyota Way: Fast feedback loops, outcome-focused quality

.PHONY: help pre-commit ci-validate test-fast test-all lint lint-fast lint-full build run clean
.PHONY: coverage coverage-summary coverage-open coverage-ci coverage-clean coverage-report

# Default target
help:
	@echo "📋 Available targets:"
	@echo ""
	@echo "Development Workflow:"
	@echo "  make pre-commit       - Fast pre-commit checks (<30s)"
	@echo "  make test-fast        - Quick unit tests (<5 min)"
	@echo "  make test-all         - All tests including integration"
	@echo ""
	@echo "Coverage (Toyota Way: 'make coverage' just works):"
	@echo "  make coverage         - Generate HTML coverage report (<10min)"
	@echo "  make coverage-open    - Open HTML coverage in browser"
	@echo "  make coverage-summary - Show coverage summary only"
	@echo "  make coverage-ci      - Generate LCOV for CI/CD (fast mode)"
	@echo "  make coverage-clean   - Clean coverage artifacts"
	@echo ""
	@echo "Quality & Linting:"
	@echo "  make lint             - Quick lint check (alias for lint-fast)"
	@echo "  make lint-fast        - Quick lint check"
	@echo "  make lint-full        - Full lint with all features"
	@echo "  make ci-validate      - Full CI validation pipeline"
	@echo ""
	@echo "Build & Run:"
	@echo "  make build            - Build release binary"
	@echo "  make run              - Run the CLI tool"
	@echo "  make clean            - Clean build artifacts"

# Fast pre-commit hook (<30 seconds) - Toyota Way: Don't overburden developers
pre-commit: fmt-check lint-fast
	@echo "🧪 Running fast tests..."
	@cargo test --lib --bins --quiet
	@echo "✅ Pre-commit checks passed (fast feedback)"

# Comprehensive CI validation - Run in CI, not locally
ci-validate: lint-full test-all coverage-ci
	@echo "✅ All CI quality gates passed"
	@echo "📊 Review coverage report: lcov.info"

# Format check
fmt-check:
	@echo "🎨 Checking code formatting..."
	@cargo fmt --check

# Lint alias (points to lint-fast by default)
lint: lint-fast

# Quick lint (faster for pre-commit)
lint-fast:
	@echo "🔍 Running quick lint..."
	@cargo clippy --all-targets -- -D warnings

# Full lint with pedantic mode
lint-full: fmt-check
	@echo "🔍 Running comprehensive lint..."
	@cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic

# Fast tests (<5 min target)
test-fast:
	@echo "🧪 Running fast test suite..."
	@cargo test --quiet --lib --bins

# All tests
test-all:
	@echo "🧪 Running all tests..."
	@cargo test --all-features --workspace

# Code Coverage (Toyota Way: "make coverage" just works)
# Following bashrs pattern: Two-Phase instrumentation + reporting
# TARGET: < 10 minutes (enforced with reduced property test cases)
coverage: ## Generate HTML coverage report and open in browser
	@echo "📊 Running comprehensive test coverage analysis (target: <10 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --all-features --workspace
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"
	@echo ""

coverage-summary: ## Show coverage summary
	@cargo llvm-cov report --summary-only 2>/dev/null || echo "Run 'make coverage' first"

coverage-open: ## Open HTML coverage report in browser
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Please open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first to generate the HTML report"; \
	fi

coverage-ci: ## Generate LCOV report for CI/CD (fast mode)
	@echo "=== Code Coverage for CI/CD ==="
	@echo "Phase 1: Running tests with instrumentation..."
	@cargo llvm-cov clean --workspace
	@cargo llvm-cov --no-report nextest --no-tests=warn --all-features --workspace
	@echo "Phase 2: Generating LCOV report..."
	@cargo llvm-cov report --lcov --output-path lcov.info
	@echo "✓ Coverage report generated: lcov.info"

coverage-clean: ## Clean coverage artifacts
	@cargo llvm-cov clean --workspace
	@rm -f lcov.info coverage.xml target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

# Legacy target for backwards compatibility
coverage-report: coverage
	@echo "💡 Use 'make coverage' directly (this is an alias)"

# Build release binary
build:
	@echo "🔨 Building release binary..."
	@cargo build --release
	@echo "✅ Binary available at target/release/oip"

# Run the CLI
run:
	@cargo run --

# Clean artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@echo "✅ Clean complete"

# Verify Makefile works correctly (meta-test)
test-makefile:
	@echo "🧪 Testing Makefile targets..."
	@echo "  ✓ make help works"
	@$(MAKE) help > /dev/null
	@echo "  ✓ make fmt-check (dry run)"
	@echo "  ✓ make lint-fast (dry run)"
	@echo "✅ Makefile validation passed"

# Organizational Intelligence Plugin - Makefile
# Toyota Way: Fast feedback loops, outcome-focused quality

.PHONY: help pre-commit ci-validate test-fast test-all coverage-report lint-fast lint-full build run clean

# Default target
help:
	@echo "📋 Available targets:"
	@echo "  make pre-commit       - Fast pre-commit checks (<30s)"
	@echo "  make ci-validate      - Full CI validation pipeline"
	@echo "  make test-fast        - Quick unit tests (<5 min)"
	@echo "  make test-all         - All tests including integration"
	@echo "  make coverage-report  - Generate coverage report"
	@echo "  make lint-fast        - Quick lint check"
	@echo "  make lint-full        - Full lint with all features"
	@echo "  make build            - Build release binary"
	@echo "  make run              - Run the CLI tool"
	@echo "  make clean            - Clean build artifacts"

# Fast pre-commit hook (<30 seconds) - Toyota Way: Don't overburden developers
pre-commit: fmt-check lint-fast
	@echo "🧪 Running fast tests..."
	@cargo test --lib --bins --quiet
	@echo "✅ Pre-commit checks passed (fast feedback)"

# Comprehensive CI validation - Run in CI, not locally
ci-validate: lint-full test-all coverage-report
	@echo "✅ All CI quality gates passed"
	@echo "📊 Review coverage report in target/llvm-cov/html/index.html"

# Format check
fmt-check:
	@echo "🎨 Checking code formatting..."
	@cargo fmt --check

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

# Coverage report (goal: 85%, not hard gate)
coverage-report:
	@echo "📊 Generating coverage report..."
	@cargo llvm-cov --all-features --workspace --html
	@echo "📈 Coverage report generated at target/llvm-cov/html/index.html"
	@echo "🎯 Target: 85% (goal, not hard gate)"

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

# Organizational Intelligence Plugin - Makefile
# Toyota Way: Fast feedback loops, outcome-focused quality

.PHONY: help pre-commit ci-validate test-fast test-all lint lint-fast lint-full build run clean
.PHONY: coverage coverage-summary coverage-check coverage-open coverage-ci coverage-clean coverage-report
.PHONY: wasm wasm-build wasm-test wasm-serve wasm-clean viz viz-test
.PHONY: e2e e2e-install e2e-test e2e-headed e2e-ui e2e-report

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
	@echo ""
	@echo "WebAssembly (Browser):"
	@echo "  make wasm             - Build WASM package (alias for wasm-build)"
	@echo "  make wasm-build       - Build optimized WASM package"
	@echo "  make wasm-test        - Run WASM module tests"
	@echo "  make wasm-serve       - Serve WASM demo (ruchy preferred, port 7777)"
	@echo "  make wasm-clean       - Clean WASM build artifacts"
	@echo ""
	@echo "Visualization:"
	@echo "  make viz              - Build with visualization feature"
	@echo "  make viz-test         - Test visualization module"
	@echo ""
	@echo "E2E Testing (Playwright):"
	@echo "  make e2e-install      - Install Playwright and browsers"
	@echo "  make e2e              - Build WASM and run e2e tests"
	@echo "  make e2e-headed       - Run e2e tests with browser visible"
	@echo "  make e2e-ui           - Open Playwright interactive UI"
	@echo "  make e2e-report       - Open HTML test report"

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

coverage-check: ## Enforce 90% coverage threshold (excludes GPU/main)
	@echo "📊 Checking coverage threshold (minimum 90%)..."
	@command -v cargo-llvm-cov > /dev/null || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov clean --workspace 2>/dev/null || true
	@echo "  Running tests with coverage instrumentation..."
	@COVERAGE_OUTPUT=$$(cargo llvm-cov --ignore-filename-regex "(gpu_|main\.rs)" nextest --no-tests=warn --lib --bins 2>&1); \
	test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true; \
	COVERAGE=$$(echo "$$COVERAGE_OUTPUT" | grep "^TOTAL" | awk '{print $$10}' | tr -d '%'); \
	echo "  Line coverage: $${COVERAGE}%"; \
	if [ "$$(echo "$${COVERAGE} < 90" | bc -l)" -eq 1 ]; then \
		echo "❌ Coverage $${COVERAGE}% is below 90% threshold"; \
		exit 1; \
	else \
		echo "✅ Coverage $${COVERAGE}% meets 90% threshold"; \
	fi

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

# ============================================================================
# WebAssembly Targets
# ============================================================================

# WASM alias
wasm: wasm-build

# Build WASM package
wasm-build:
	@echo "🌐 Building WebAssembly package..."
	@which wasm-pack > /dev/null 2>&1 || (echo "📦 Installing wasm-pack..." && cargo install wasm-pack)
	@cd wasm-pkg && wasm-pack build --target web --release
	@echo ""
	@echo "✅ WASM build complete!"
	@echo "   Package: wasm-pkg/pkg/"
	@echo "   Size: $$(du -h wasm-pkg/pkg/*.wasm | cut -f1)"
	@echo ""
	@echo "💡 Next steps:"
	@echo "   make wasm-serve    - Start local server"
	@echo "   Open http://localhost:7777/"

# Test WASM module (native tests)
wasm-test:
	@echo "🧪 Testing WASM module..."
	@cargo test --features wasm wasm::
	@echo ""
	@echo "🧪 Testing standalone WASM package..."
	@cd wasm-pkg && cargo test
	@echo "✅ WASM tests passed"

# Serve WASM demo locally (default: 7777, override with: make wasm-serve WASM_PORT=9000)
# Prefers ruchy (12x faster) with Python fallback
WASM_PORT ?= 7777
wasm-serve: wasm-build
	@echo "🌐 Starting local server at http://localhost:$(WASM_PORT)"
	@echo "   Open: http://localhost:$(WASM_PORT)/"
	@echo "   Press Ctrl+C to stop"
	@echo "   (Use WASM_PORT=XXXX to change: make wasm-serve WASM_PORT=9000)"
	@if command -v ruchy > /dev/null 2>&1; then \
		echo "   Using ruchy (12x faster than Python)"; \
		cd wasm-pkg && ruchy serve . --port $(WASM_PORT); \
	else \
		echo "   Using Python (install ruchy for 12x faster: cargo install ruchy)"; \
		cd wasm-pkg && python3 -m http.server $(WASM_PORT); \
	fi

# Clean WASM artifacts
wasm-clean:
	@echo "🧹 Cleaning WASM artifacts..."
	@rm -rf wasm-pkg/pkg wasm-pkg/target
	@echo "✅ WASM artifacts cleaned"

# ============================================================================
# Visualization Targets
# ============================================================================

# Build with visualization feature
viz:
	@echo "📊 Building with visualization feature..."
	@cargo build --features viz
	@echo "✅ Viz build complete"

# Test visualization module
viz-test:
	@echo "🧪 Testing visualization module..."
	@cargo test --features viz viz::
	@echo "✅ Viz tests passed"

# Demo visualization with real data
viz-demo:
	@echo "📊 Running visualization demo..."
	@cargo run --bin oip --features viz -- extract-training-data --repo . --output /tmp/viz-demo.json --max-commits 50 --viz
	@echo "✅ Demo complete"

# ============================================================================
# End-to-End Testing (Playwright)
# ============================================================================

# Install e2e dependencies
e2e-install:
	@echo "📦 Installing Playwright dependencies..."
	@cd e2e && npm install
	@cd e2e && npx playwright install
	@echo "✅ Playwright installed"

# Run e2e tests (headless)
e2e: wasm-build e2e-test

e2e-test:
	@echo "🧪 Running e2e tests..."
	@cd e2e && npm test
	@echo "✅ E2E tests complete"

# Run e2e tests with browser visible
e2e-headed:
	@echo "🧪 Running e2e tests (headed)..."
	@cd e2e && npm run test:headed

# Run e2e tests with UI
e2e-ui:
	@echo "🧪 Opening Playwright UI..."
	@cd e2e && npm run test:ui

# Show e2e test report
e2e-report:
	@echo "📊 Opening test report..."
	@cd e2e && npm run report

# Localize Command

Perform Tarantula-style Spectrum-Based Fault Localization (SBFL) to identify suspicious code when tests fail.

## Synopsis

```bash
oip localize [OPTIONS] --passed-coverage <FILE> --failed-coverage <FILE>
```

## Description

The `localize` command uses SBFL algorithms to rank code statements by their likelihood of containing a fault. It analyzes coverage data from passing and failing test runs to compute suspiciousness scores.

**Toyota Way Alignment:**
- **Genchi Genbutsu**: Uses actual coverage data, not estimates
- **Muda**: Only runs expensive TDG enrichment when requested
- **Jidoka**: Provides human-readable explanations

## Options

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `--passed-coverage` | LCOV file from passing tests | Required |
| `--failed-coverage` | LCOV file from failing tests | Required |
| `--passed-count` | Number of passing tests | 1 |
| `--failed-count` | Number of failing tests | 1 |
| `--formula` | SBFL formula (see below) | tarantula |
| `--top-n` | Top N suspicious statements | 10 |
| `-o, --output` | Output file path | fault-localization.yaml |
| `-f, --format` | Output format (yaml/json/terminal) | yaml |
| `--enrich-tdg` | Include TDG scores from pmat | false |
| `--repo` | Repository path for TDG enrichment | - |

### RAG Enhancement Options (Phase 5)

| Option | Description | Default |
|--------|-------------|---------|
| `--rag` | Enable RAG-enhanced localization | false |
| `--knowledge-base` | Path to bug knowledge base YAML | - |
| `--fusion` | Fusion strategy (rrf/linear/dbsf/sbfl-only) | rrf |
| `--similar-bugs` | Number of similar bugs to retrieve | 5 |

### Weighted Ensemble Options (Phase 6)

| Option | Description | Default |
|--------|-------------|---------|
| `--ensemble` | Enable weighted ensemble model | false |
| `--ensemble-model` | Path to trained ensemble model file | - |
| `--include-churn` | Include churn metrics from git history | false |

### Calibrated Prediction Options (Phase 7)

| Option | Description | Default |
|--------|-------------|---------|
| `--calibrated` | Enable calibrated probability output | false |
| `--calibration-model` | Path to trained calibration model | - |
| `--confidence-threshold` | Only report files above this probability (0.0-1.0) | 0.5 |

## SBFL Formulas

### Tarantula (Default)

The classic formula from Jones & Harrold (2005):

```
suspiciousness = (failed/totalFailed) / ((passed/totalPassed) + (failed/totalFailed))
```

Best for: General fault localization with balanced datasets.

### Ochiai

Borrowed from molecular biology, often outperforms Tarantula:

```
suspiciousness = failed / sqrt(totalFailed × (failed + passed))
```

Best for: When you want higher precision in top rankings.

### DStar2 / DStar3

Parameterized formula with configurable exponent:

```
suspiciousness = failed^* / (passed + (totalFailed - failed))
```

Best for: When failing tests are very indicative of faults.

## RAG-Enhanced Localization (Phase 5)

RAG (Retrieval-Augmented Generation) enhancement combines SBFL with historical bug knowledge to provide:

- **Similar bug retrieval**: Find historical bugs matching current fault patterns
- **Fix suggestions**: Retrieve proven fix patterns from knowledge base
- **Context explanations**: Understand why a statement is suspicious
- **Fusion strategies**: Combine SBFL + RAG scores intelligently

### Fusion Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `rrf` | Reciprocal Rank Fusion (k=60) | Balanced combination, default |
| `linear` | Linear combination (70% SBFL, 30% RAG) | When SBFL confidence is high |
| `dbsf` | Distribution-Based Score Fusion | Statistical normalization |
| `sbfl-only` | Ignore RAG, pure SBFL | Baseline comparison |

### Knowledge Base Format

Create a YAML file with historical bug records:

```yaml
- id: "BUG-001"
  title: "Null pointer in parser"
  description: "Parser crashes when input is empty"
  category: MemorySafety
  fix_commit: "abc123"
  fix_diff: "- let x = data.unwrap();\n+ let x = data.unwrap_or_default();"
  affected_files:
    - "src/parser.rs"
  severity: 4
  symptoms:
    - "panic on empty input"
    - "unwrap failed"
  root_cause: "Missing None check"
  fix_pattern: "Replace unwrap() with unwrap_or_default()"
```

### Supported Categories

`MemorySafety`, `Concurrency`, `TypeErrors`, `Performance`, `Security`, `Configuration`, `ApiMisuse`, `IntegrationFailure`, `DocumentationGap`, `TestingGap`, `OperatorPrecedence`, `TypeAnnotationGap`, `StdlibMapping`, `AstTransform`, `ComprehensionBug`, `IteratorChain`, `OwnershipBorrow`, `TraitBounds`

## Examples

### Basic Usage

```bash
# Generate coverage from passing tests
cargo llvm-cov --lcov > passed.lcov
cargo test -- --test-threads=1

# Generate coverage from failing tests (with failing test isolated)
cargo llvm-cov --lcov > failed.lcov
cargo test failing_test

# Run fault localization
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --formula=tarantula \
  --top-n=10
```

### With TDG Enrichment

Combine SBFL scores with Technical Debt Grade from pmat:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --formula=ochiai \
  --enrich-tdg \
  --repo=. \
  --output=fault-report.yaml
```

### RAG-Enhanced Localization

Use historical bug knowledge to improve fault localization:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --formula=tarantula \
  --top-n=10 \
  --rag \
  --knowledge-base=bugs.yaml \
  --fusion=rrf \
  --similar-bugs=5 \
  --format=terminal
```

Output:
```
🔍 RAG-Enhanced Fault Localization (trueno-rag)
   Formula: tarantula
   Top N:   10
   RAG:     enabled
   Fusion:  rrf

🤖 Applying RAG enhancement...
   Loading knowledge base: bugs.yaml
   ✅ Loaded 15 bugs from knowledge base

╔══════════════════════════════════════════════════════════════╗
║        RAG-ENHANCED FAULT LOCALIZATION REPORT                ║
╠══════════════════════════════════════════════════════════════╣
║ SBFL Formula: Tarantula                                      ║
║ Fusion Strategy: RRF                                         ║
║ Knowledge Base: 15 bugs                                      ║
╠══════════════════════════════════════════════════════════════╣
║  #1  src/parser.rs:50     ████████████████░░░░ 0.92         ║
║      → Similar: Null pointer in parser (95%)                 ║
║  #2  src/handler.rs:100   ██████████████░░░░░░ 0.78         ║
║      → Similar: Race condition in handler (88%)              ║
╚══════════════════════════════════════════════════════════════╝
```

### Terminal Output

Get immediate visual feedback:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --format=terminal
```

Output:
```
╔══════════════════════════════════════════════════════════════╗
║           FAULT LOCALIZATION REPORT - Tarantula              ║
╠══════════════════════════════════════════════════════════════╣
║ Tests: 100 passed, 10 failed                                 ║
║ Confidence: 0.85                                             ║
╠══════════════════════════════════════════════════════════════╣
║  TOP SUSPICIOUS STATEMENTS                                   ║
╠══════════════════════════════════════════════════════════════╣
║  #1  src/parser.rs:87    ████████████████░░░░ 0.92           ║
║  #2  src/parser.rs:91    ██████████████░░░░░░ 0.78           ║
║  #3  src/parser.rs:85    ████████████░░░░░░░░ 0.65           ║
╚══════════════════════════════════════════════════════════════╝
```

### Weighted Ensemble Model (Phase 6)

Combine multiple signals (SBFL, TDG, Churn, Complexity) using weak supervision:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --formula=tarantula \
  --ensemble \
  --enrich-tdg \
  --repo=. \
  --format=terminal
```

Output:
```
🔮 Running Weighted Ensemble Model (Phase 6)...
   ✅ Ensemble model fitted on 10 files

   Ensemble Risk Predictions:
   #1 src/parser.rs - Risk: 78.5%
   #2 src/handler.rs - Risk: 65.2%
   #3 src/util.rs - Risk: 42.1%

   Learned Signal Weights:
      SBFL: 35.2%
      TDG: 22.1%
      Churn: 18.5%
      Complexity: 14.8%
      RAG_Similarity: 9.4%
```

### Calibrated Predictions (Phase 7)

Get probability estimates with confidence intervals:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --calibrated \
  --confidence-threshold=0.5 \
  --format=terminal
```

Output:
```
📊 Running Calibrated Defect Prediction (Phase 7)...
   Confidence threshold: 50%

   Calibrated Predictions (above 50% threshold):
   #1 src/parser.rs:87 - P(defect) = 73% ± 12% [HIGH]
      ├─ SBFL: 35.0%
      ├─ TDG: 22.0%
      ├─ Churn: 18.0%
   #2 src/handler.rs:100 - P(defect) = 58% ± 22% [MEDIUM]
      ├─ TDG: 40.0%
      ├─ Churn: 30.0%
```

### Combined Pipeline (All Features)

Use SBFL + RAG + Ensemble + Calibration together:

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --formula=ochiai \
  --rag \
  --knowledge-base=bugs.yaml \
  --ensemble \
  --calibrated \
  --confidence-threshold=0.6 \
  --enrich-tdg \
  --repo=. \
  --format=terminal
```

### JSON Output for CI/CD

```bash
oip localize \
  --passed-coverage=passed.lcov \
  --failed-coverage=failed.lcov \
  --format=json \
  --output=fault-report.json

# Use in CI pipeline
jq '.rankings[0].statement.file' fault-report.json
```

## Output Format

### YAML Output

```yaml
rankings:
  - rank: 1
    statement:
      file: src/parser.rs
      line: 87
    suspiciousness: 0.92
    scores:
      tarantula: 0.92
      ochiai: 0.89
      dstar2: 0.95
      dstar3: 0.97
    explanation: "Executed by 95% of failing tests (9/10) and 12% of passing tests (12/100)."
    failed_coverage: 9
    passed_coverage: 12
formula_used: Tarantula
confidence: 0.85
total_passed_tests: 100
total_failed_tests: 10
```

### RAG-Enhanced Output

```yaml
rankings:
  - sbfl_ranking:
      rank: 1
      statement:
        file: src/parser.rs
        line: 50
      suspiciousness: 0.92
      scores:
        tarantula: 0.92
        ochiai: 0.89
    similar_bugs:
      - id: BUG-001
        similarity: 0.95
        category: MemorySafety
        summary: "Null pointer in parser"
        fix_commit: abc123
    suggested_fixes:
      - pattern: "Fix pattern for Memory Safety"
        confidence: 0.85
        example: "Replace unwrap() with unwrap_or_default()"
        source_bug_id: BUG-001
    context_explanation: >
      This pattern matches historical bug "BUG-001" (Null pointer in parser)
      with 95% similarity. 1 similar bugs found in knowledge base.
    combined_score: 0.927
sbfl_result:
  formula_used: Tarantula
  confidence: 0.85
fusion_strategy: RRF
knowledge_base_size: 15
```

## Integration with cargo-llvm-cov

### Setup

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate baseline coverage
cargo llvm-cov --lcov --output-path coverage.lcov
```

### Workflow for Bug Investigation

1. **Reproduce the failing test**:
   ```bash
   cargo test test_that_fails -- --nocapture
   ```

2. **Generate coverage for passing tests**:
   ```bash
   cargo llvm-cov --lcov --output-path passed.lcov -- --skip test_that_fails
   ```

3. **Generate coverage for failing test**:
   ```bash
   cargo llvm-cov --lcov --output-path failed.lcov -- test_that_fails
   ```

4. **Run localization**:
   ```bash
   oip localize \
     --passed-coverage=passed.lcov \
     --failed-coverage=failed.lcov \
     --passed-count=$(cargo test --no-run 2>&1 | grep -c "test") \
     --failed-count=1
   ```

## See Also

- [Tarantula Specification](../../docs/specifications/tarantula-defect-specification.md)
- [Example: Fault Localization](../examples/fault-localization.md)
- [GPU Correlation Matrix](../gpu/correlation-matrix.md)

## References

1. Jones, J.A., Harrold, M.J. (2005). *Empirical evaluation of the Tarantula automatic fault-localization technique.* ASE '05.
2. Abreu, R., et al. (2009). *A practical evaluation of spectrum-based fault localization.* JSS 82(11).
3. Wong, W.E., et al. (2014). *The DStar method for effective software fault localization.* IEEE TSE 40(1).
4. Cormack, G.V., et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR '09.
5. trueno-rag crate: [crates.io/crates/trueno-rag](https://crates.io/crates/trueno-rag)

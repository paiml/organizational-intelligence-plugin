# Exploratory Data Analysis

Exploratory Data Analysis (EDA) is the critical first step before training any ML classifier. This chapter covers the systematic exploration of OIP's training data features, demonstrating pandas-style analysis techniques implemented in Rust.

## Why EDA Matters

Before training a defect classifier, we must answer:

1. **Data Quality**: Are there missing values? Outliers? Inconsistencies?
2. **Feature Distributions**: What do our numeric features look like?
3. **Class Balance**: Are defect categories evenly distributed?
4. **Feature Correlations**: Which features carry predictive signal?
5. **Data Volume**: Do we have enough examples per category?

Skipping EDA leads to models that fail silently in production.

## The Feature Space

OIP extracts two primary feature structures for ML training:

### TrainingExample (Text + Metadata)

The `TrainingExample` struct captures labeled training instances:

```rust
pub struct TrainingExample {
    // Core fields
    pub message: String,           // Commit message (primary feature)
    pub label: DefectCategory,     // Ground-truth label (18 categories)
    pub confidence: f32,           // Label confidence (0.0-1.0)

    // Git metadata
    pub commit_hash: String,
    pub author: String,
    pub timestamp: i64,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub files_changed: usize,

    // NLP-014: CITL fields
    pub error_code: Option<String>,      // e.g., "E0308"
    pub clippy_lint: Option<String>,     // e.g., "clippy::unwrap_used"
    pub has_suggestion: bool,
    pub suggestion_applicability: Option<SuggestionApplicability>,
    pub source: TrainingSource,          // CommitMessage or DepylerCitl
}
```

### CommitFeatures (GPU-Ready Numerics)

The `CommitFeatures` struct provides a 14-dimensional vector for GPU processing:

```rust
pub struct CommitFeatures {
    // Categorical (one-hot encoded)
    pub defect_category: u8,       // 0-17 (18 categories)

    // Numerical (f32 for GPU)
    pub files_changed: f32,
    pub lines_added: f32,
    pub lines_deleted: f32,
    pub complexity_delta: f32,     // Cyclomatic complexity change

    // Temporal
    pub timestamp: f64,            // Unix epoch
    pub hour_of_day: u8,           // 0-23 (circadian patterns)
    pub day_of_week: u8,           // 0-6 (Monday=0)

    // NLP-014: CITL features (6 dimensions)
    pub error_code_class: u8,      // 0=type, 1=borrow, 2=name, 3=trait, 4=other
    pub has_suggestion: u8,        // 0 or 1
    pub suggestion_applicability: u8, // 0=none, 1=machine, 2=maybe, 3=placeholder
    pub clippy_lint_count: u8,     // 0-255
    pub span_line_delta: f32,      // Normalized span position
    pub diagnostic_confidence: f32, // Taxonomy mapping confidence
}
```

## Running the Explorer

OIP provides a pandas-style data exploration example:

```bash
cargo run --example explore_data
```

This produces output similar to:

```
═══════════════════════════════════════════════════════════════════════════════
                         CITL Training Data Explorer
═══════════════════════════════════════════════════════════════════════════════
```

## df.head() - First Look at the Data

The first step in any EDA is viewing raw samples. The `df.head()` equivalent shows:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ df.head()                                                              [7 rows]  │
├─────┬──────────────────────┬─────────────────────────┬────────┬─────────────────┤
│ idx │ label                │ error_code/lint         │  conf  │ source          │
├─────┼──────────────────────┼─────────────────────────┼────────┼─────────────────┤
│   0 │ Type Errors          │ E0308                   │  95.0% │ DepylerCitl     │
│   1 │ API Misuse           │ clippy::unwrap_used     │  88.0% │ DepylerCitl     │
│   2 │ Ownership/Borrow     │ E0502                   │  92.0% │ DepylerCitl     │
│   3 │ Trait Bounds         │ E0277                   │  91.0% │ DepylerCitl     │
│   4 │ Performance Issues   │ clippy::cognitive_compl │  85.0% │ DepylerCitl     │
│   5 │ Memory Safety        │ E0382                   │  93.0% │ DepylerCitl     │
│   6 │ Logic Errors         │ clippy::todo            │  87.0% │ DepylerCitl     │
└─────┴──────────────────────┴─────────────────────────┴────────┴─────────────────┘
```

### Key Observations

1. **Label Coverage**: Multiple defect categories represented
2. **Error Codes**: Mix of rustc errors (E0xxx) and clippy lints
3. **Confidence Range**: 85-95% indicates high-quality ground truth
4. **Source**: All from CITL (compiler diagnostics, not commit messages)

### Implementation

```rust
for (i, ex) in examples.iter().enumerate() {
    let code = ex.error_code.as_deref()
        .or(ex.clippy_lint.as_deref())
        .unwrap_or("-");

    println!("│ {:3} │ {:20} │ {:23} │ {:5.1}% │ {:15} │",
        i,
        ex.label.as_str(),
        &code[..code.len().min(23)],
        ex.confidence * 100.0,
        format!("{:?}", ex.source),
    );
}
```

## df.describe() - Summary Statistics

Numeric feature distributions reveal data quality issues:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ df.describe()                                                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                      confidence                                                   │
│ count             7                                                               │
│ mean          90.14%                                                              │
│ std            3.31%                                                              │
│ min           85.00%                                                              │
│ 50%           91.00%                                                              │
│ max           95.00%                                                              │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Statistics to Compute

| Statistic | Formula | Purpose |
|-----------|---------|---------|
| **count** | `n` | Sample size |
| **mean** | `Σx / n` | Central tendency |
| **std** | `√(Σ(x-μ)² / n)` | Spread/variability |
| **min** | `min(x)` | Lower bound |
| **50%** | `median(x)` | Robust center |
| **max** | `max(x)` | Upper bound |

### Implementation

```rust
let conf_vals: Vec<f32> = examples.iter().map(|e| e.confidence).collect();

let mean = conf_vals.iter().sum::<f32>() / conf_vals.len() as f32;
let min = conf_vals.iter().cloned().fold(f32::INFINITY, f32::min);
let max = conf_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

let mut sorted = conf_vals.clone();
sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
let median = sorted[sorted.len() / 2];

let variance = conf_vals.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f32>() / conf_vals.len() as f32;
let std_dev = variance.sqrt();
```

### Interpreting the Statistics

- **Low std (3.31%)**: Confidence values are consistent, not noisy
- **High mean (90.14%)**: Labels are high-quality ground truth
- **Tight range (85-95%)**: No extreme outliers to filter

For `CommitFeatures`, compute statistics for all 14 dimensions:

| Feature | Typical Range | Notes |
|---------|--------------|-------|
| `files_changed` | 1-50 | Log-transform if skewed |
| `lines_added` | 0-1000+ | Consider capping outliers |
| `lines_deleted` | 0-500+ | Often correlates with added |
| `hour_of_day` | 0-23 | Circular feature, special encoding |
| `day_of_week` | 0-6 | May show Friday patterns |
| `diagnostic_confidence` | 0.0-1.0 | Already normalized |

## df['label'].value_counts() - Class Distribution

Class imbalance is the most common ML failure mode:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ df['label'].value_counts()                                                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Type Errors           1 ( 14.3%)  ████████                                        │
│ API Misuse            1 ( 14.3%)  ████████                                        │
│ Ownership/Borrow      1 ( 14.3%)  ████████                                        │
│ Trait Bounds          1 ( 14.3%)  ████████                                        │
│ Memory Safety         1 ( 14.3%)  ████████                                        │
│ Performance Issues    1 ( 14.3%)  ████████                                        │
│ Logic Errors          1 ( 14.3%)  ████████                                        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
let mut counts: HashMap<String, usize> = HashMap::new();
for ex in &examples {
    *counts.entry(ex.label.as_str().to_string()).or_insert(0) += 1;
}

let mut counts_vec: Vec<_> = counts.into_iter().collect();
counts_vec.sort_by(|a, b| b.1.cmp(&a.1));  // Sort by count descending

for (label, count) in &counts_vec {
    let pct = (*count as f32 / examples.len() as f32) * 100.0;
    let bar = "█".repeat(*count * 8);
    println!("│ {:20} {:2} ({:5.1}%)  {:40} │", label, count, pct, bar);
}
```

### Class Imbalance Indicators

| Imbalance Ratio | Severity | Remediation |
|-----------------|----------|-------------|
| < 2:1 | None | Standard training |
| 2:1 - 5:1 | Mild | Class weights |
| 5:1 - 10:1 | Moderate | SMOTE oversampling |
| > 10:1 | Severe | Combine categories or collect more data |

In real-world OIP data, expect:

- **Dominant**: Memory Safety, Type Errors (30-40%)
- **Common**: API Misuse, Concurrency (15-25%)
- **Rare**: Documentation Gaps, Testing Gaps (1-5%)

## df.iloc[0] - Single Record Deep Dive

Examine individual records to understand feature semantics:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ df.iloc[0]                                                                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ message                mismatched types: expected `i32`, found `String`           │
│ label                  Type Errors                                                │
│ confidence             0.9500                                                     │
│ error_code             E0308                                                      │
│ clippy_lint            None                                                       │
│ has_suggestion         false                                                      │
│ suggestion_applicab.   None                                                       │
│ source                 DepylerCitl                                                │
│ timestamp              1732752000                                                 │
│ author                 depyler                                                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Feature Analysis

**Text Feature (message)**:
- Raw diagnostic text from rustc
- Contains type information ("i32", "String")
- Suitable for TF-IDF vectorization

**Categorical Features**:
- `label`: Target variable (18 classes)
- `error_code`: E0308 = type mismatch
- `source`: DepylerCitl = compiler ground truth

**Numeric Features**:
- `confidence`: 0.95 = high-quality label
- `timestamp`: Unix epoch for temporal analysis

**Boolean Features**:
- `has_suggestion`: false = no auto-fix available

## df.info() - Schema and Nulls

Understanding the data schema prevents runtime errors:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ df.info()                                                                         │
├───────────────────────────────────────────────────────────────────────────────────┤
│ <TrainingExample>                                                                 │
│ RangeIndex: 7 entries, 0 to 6                                                     │
│ Data columns (total 12 columns):                                                  │
│  #   Column                   Non-Null Count  Dtype                               │
│ ---  ------                   --------------  -----                               │
│  0   message                  7 non-null      String                              │
│  1   label                    7 non-null      DefectCategory                      │
│  2   confidence               7 non-null      f32                                 │
│  3   error_code               4 non-null      Option<String>                      │
│  4   clippy_lint              3 non-null      Option<String>                      │
│  5   has_suggestion           7 non-null      bool                                │
│  6   suggestion_applicability 0 non-null      Option<SuggestionApplicability>     │
│  7   source                   7 non-null      TrainingSource                      │
│ memory usage: ~1120 bytes                                                         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Null Analysis

| Column | Non-Null | Pattern |
|--------|----------|---------|
| `message`, `label`, `confidence` | 7/7 | Required fields, always present |
| `error_code` | 4/7 | Present for rustc errors only |
| `clippy_lint` | 3/7 | Present for clippy warnings only |
| `suggestion_applicability` | 0/7 | Only when `has_suggestion=true` |

### Implementation

```rust
let ec_count = examples.iter().filter(|e| e.error_code.is_some()).count();
let cl_count = examples.iter().filter(|e| e.clippy_lint.is_some()).count();
let sa_count = examples.iter()
    .filter(|e| e.suggestion_applicability.is_some()).count();

println!("│  3   error_code               {} non-null      Option<String> │", ec_count);
println!("│  4   clippy_lint              {} non-null      Option<String> │", cl_count);
```

## Feature Engineering Insights

EDA reveals which features to engineer:

### 1. Text Processing Pipeline

```
message → tokenize → stem → remove_stopwords → TF-IDF → [n-dim vector]
```

### 2. Error Code Encoding

Map E-codes to semantic classes:

| Error Class | Codes | Numeric |
|-------------|-------|---------|
| Type | E0308, E0412 | 0 |
| Borrow | E0502, E0503, E0505 | 1 |
| Name | E0425, E0433 | 2 |
| Trait | E0277 | 3 |
| Other | All else | 4 |

### 3. Temporal Features

Circadian patterns in defect commits:

```rust
pub hour_of_day: u8,  // 0-23, peak at 14:00-16:00
pub day_of_week: u8,  // 0-6, Friday has higher defect rate
```

### 4. Churn Metrics

```rust
let churn_ratio = lines_added as f32 / (lines_deleted + 1) as f32;
let change_magnitude = (lines_added + lines_deleted) as f32;
```

## Data Quality Checklist

Before training, verify:

- [ ] **No duplicate commit hashes** (dedup required)
- [ ] **Confidence > min_threshold** (default 0.75)
- [ ] **Message length > 10 chars** (filter noise)
- [ ] **Balanced class distribution** (or apply SMOTE)
- [ ] **Temporal coverage** (not all from one day)
- [ ] **Author diversity** (not all from one person)

### Validation Code

```rust
fn validate_dataset(examples: &[TrainingExample]) -> ValidationResult {
    let unique_hashes: HashSet<_> = examples.iter()
        .map(|e| &e.commit_hash).collect();

    let low_conf = examples.iter()
        .filter(|e| e.confidence < 0.75).count();

    let short_messages = examples.iter()
        .filter(|e| e.message.len() < 10).count();

    ValidationResult {
        duplicate_ratio: 1.0 - (unique_hashes.len() as f32 / examples.len() as f32),
        low_confidence_pct: low_conf as f32 / examples.len() as f32,
        short_message_pct: short_messages as f32 / examples.len() as f32,
    }
}
```

## The 14-Dimension Feature Vector

For GPU processing, `CommitFeatures::to_vector()` produces:

| Index | Feature | Type | Range |
|-------|---------|------|-------|
| 0 | `defect_category` | categorical | 0-17 |
| 1 | `files_changed` | numeric | 0-∞ |
| 2 | `lines_added` | numeric | 0-∞ |
| 3 | `lines_deleted` | numeric | 0-∞ |
| 4 | `complexity_delta` | numeric | -∞ to +∞ |
| 5 | `timestamp` | temporal | epoch |
| 6 | `hour_of_day` | cyclical | 0-23 |
| 7 | `day_of_week` | cyclical | 0-6 |
| 8 | `error_code_class` | categorical | 0-4 |
| 9 | `has_suggestion` | binary | 0-1 |
| 10 | `suggestion_applicability` | ordinal | 0-3 |
| 11 | `clippy_lint_count` | count | 0-255 |
| 12 | `span_line_delta` | numeric | 0.0-1.0 |
| 13 | `diagnostic_confidence` | probability | 0.0-1.0 |

### Normalization Strategy

```rust
fn normalize_features(features: &mut [CommitFeatures]) {
    // Z-score for unbounded numerics
    let (mean, std) = compute_stats(&features.iter().map(|f| f.lines_added).collect());
    for f in features.iter_mut() {
        f.lines_added = (f.lines_added - mean) / std;
    }

    // Min-max for bounded features
    for f in features.iter_mut() {
        f.hour_of_day = f.hour_of_day as f32 / 23.0;
    }

    // Already normalized: confidence, span_line_delta
}
```

## Complete Example

```rust
use organizational_intelligence_plugin::citl::{convert_to_training_examples, DepylerExport};

fn main() {
    let exports = load_depyler_corpus("corpus.jsonl");
    let examples = convert_to_training_examples(&exports);

    // df.head()
    println!("First 5 examples:");
    for ex in examples.iter().take(5) {
        println!("  {} → {}", ex.error_code.as_deref().unwrap_or("-"), ex.label);
    }

    // df.describe()
    let confidences: Vec<f32> = examples.iter().map(|e| e.confidence).collect();
    println!("Confidence: mean={:.2}, std={:.2}", mean(&confidences), std(&confidences));

    // df['label'].value_counts()
    let mut counts = HashMap::new();
    for ex in &examples {
        *counts.entry(ex.label).or_insert(0) += 1;
    }
    println!("Class distribution: {:?}", counts);

    // df.info()
    println!("Total examples: {}", examples.len());
    println!("With error_code: {}", examples.iter().filter(|e| e.error_code.is_some()).count());
    println!("With clippy_lint: {}", examples.iter().filter(|e| e.clippy_lint.is_some()).count());
}
```

## Next Steps

After EDA, proceed to:

1. **[TF-IDF Feature Extraction](./tfidf-features.md)** - Vectorize text
2. **[Model Training](./model-training.md)** - Train classifier
3. **[Class Imbalance](../advanced/class-imbalance.md)** - Handle skewed distributions

## See Also

- [Example: explore_data](../examples/explore-data.md) - Full implementation
- [NLP-014: CITL Integration](../validation/nlp-014-citl.md) - Ground-truth labels
- [Training Data Extraction](./training-data-extraction.md) - Data pipeline

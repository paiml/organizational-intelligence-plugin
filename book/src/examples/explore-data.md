# Example: Explore Training Data

This example demonstrates pandas-style exploratory data analysis for CITL training data in Rust.

## Running the Example

```bash
cargo run --example explore_data
```

## What It Demonstrates

1. **df.head()** - Tabular view of training examples
2. **df.describe()** - Summary statistics (mean, std, min, median, max)
3. **df['label'].value_counts()** - Class distribution with ASCII bar chart
4. **df.iloc[0]** - Full record inspection
5. **df.info()** - Schema, dtypes, and null counts

## Sample Output

```
═══════════════════════════════════════════════════════════════════════════════
                         CITL Training Data Explorer
═══════════════════════════════════════════════════════════════════════════════

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

✅ Data exploration complete!
```

## Code Walkthrough

### Creating Sample Data

```rust
use organizational_intelligence_plugin::citl::{convert_to_training_examples, DepylerExport};

let exports = vec![
    DepylerExport {
        source_file: "src/parser.rs".into(),
        error_code: Some("E0308".into()),
        clippy_lint: None,
        level: "error".into(),
        message: "mismatched types: expected `i32`, found `String`".into(),
        oip_category: None,
        confidence: 0.95,
        span: None,
        suggestion: None,
        timestamp: 1732752000,
        depyler_version: "3.21.0".into(),
    },
    // ... more exports
];

let examples = convert_to_training_examples(&exports);
```

### Computing Statistics

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

### Value Counts with Bar Chart

```rust
use std::collections::HashMap;

let mut counts: HashMap<String, usize> = HashMap::new();
for ex in &examples {
    *counts.entry(ex.label.as_str().to_string()).or_insert(0) += 1;
}

let mut counts_vec: Vec<_> = counts.into_iter().collect();
counts_vec.sort_by(|a, b| b.1.cmp(&a.1));

for (label, count) in &counts_vec {
    let pct = (*count as f32 / examples.len() as f32) * 100.0;
    let bar = "█".repeat(*count * 8);
    println!("│ {:20} {:2} ({:5.1}%)  {:40} │", label, count, pct, bar);
}
```

### Null Count Analysis

```rust
let ec_count = examples.iter().filter(|e| e.error_code.is_some()).count();
let cl_count = examples.iter().filter(|e| e.clippy_lint.is_some()).count();
let sa_count = examples.iter()
    .filter(|e| e.suggestion_applicability.is_some()).count();

println!("error_code:   {} non-null", ec_count);
println!("clippy_lint:  {} non-null", cl_count);
println!("suggestion:   {} non-null", sa_count);
```

## Key Insights

### Data Quality Indicators

| Metric | Good | Warning | Action |
|--------|------|---------|--------|
| Confidence mean | > 85% | 70-85% | Review low-conf samples |
| Confidence std | < 10% | 10-20% | Check for label noise |
| Class balance | < 5:1 | 5:1-10:1 | Apply SMOTE |
| Null ratio | < 10% | 10-30% | Handle missing values |

### What to Look For

1. **Skewed Distributions**: Apply log transform
2. **Missing Values**: Impute or filter
3. **Class Imbalance**: Oversample minority classes
4. **Outliers**: Cap or remove extreme values
5. **Temporal Gaps**: Check for representative coverage

## Related Examples

- [CITL Import](./citl-import.md) - Loading training data
- [Classify Defects](./classify-defects.md) - Using the classifier
- [Train Custom Model](./train-custom-model.md) - Training pipeline

## See Also

- [Exploratory Data Analysis](../ml-pipeline/exploratory-data-analysis.md) - Full EDA guide
- [Training Data Extraction](../ml-pipeline/training-data-extraction.md) - Data pipeline
- [Class Imbalance](../advanced/class-imbalance.md) - Handling skewed data

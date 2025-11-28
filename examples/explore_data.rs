// Example: Explore CITL training data (df.head() / df.describe() style)
//
// Usage:
//   cargo run --example explore_data

use organizational_intelligence_plugin::citl::{convert_to_training_examples, DepylerExport};
use std::collections::HashMap;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                         CITL Training Data Explorer                            ");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    // Create sample CITL exports (simulating depyler corpus)
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
        DepylerExport {
            source_file: "src/main.rs".into(),
            error_code: None,
            clippy_lint: Some("clippy::unwrap_used".into()),
            level: "warning".into(),
            message: "used `unwrap()` on `Option` value".into(),
            oip_category: None,
            confidence: 0.88,
            span: None,
            suggestion: None,
            timestamp: 1732752001,
            depyler_version: "3.21.0".into(),
        },
        DepylerExport {
            source_file: "src/lib.rs".into(),
            error_code: Some("E0502".into()),
            clippy_lint: None,
            level: "error".into(),
            message: "cannot borrow `self.data` as mutable because it is also borrowed".into(),
            oip_category: None,
            confidence: 0.92,
            span: None,
            suggestion: None,
            timestamp: 1732752002,
            depyler_version: "3.21.0".into(),
        },
        DepylerExport {
            source_file: "src/api.rs".into(),
            error_code: Some("E0277".into()),
            clippy_lint: None,
            level: "error".into(),
            message: "the trait bound `MyType: Send` is not satisfied".into(),
            oip_category: None,
            confidence: 0.91,
            span: None,
            suggestion: None,
            timestamp: 1732752003,
            depyler_version: "3.21.0".into(),
        },
        DepylerExport {
            source_file: "src/util.rs".into(),
            error_code: None,
            clippy_lint: Some("clippy::cognitive_complexity".into()),
            level: "warning".into(),
            message: "this function has too high cognitive complexity (28/25)".into(),
            oip_category: None,
            confidence: 0.85,
            span: None,
            suggestion: None,
            timestamp: 1732752004,
            depyler_version: "3.21.0".into(),
        },
        DepylerExport {
            source_file: "src/handler.rs".into(),
            error_code: Some("E0382".into()),
            clippy_lint: None,
            level: "error".into(),
            message: "use of moved value: `data`".into(),
            oip_category: None,
            confidence: 0.93,
            span: None,
            suggestion: None,
            timestamp: 1732752005,
            depyler_version: "3.21.0".into(),
        },
        DepylerExport {
            source_file: "src/config.rs".into(),
            error_code: None,
            clippy_lint: Some("clippy::todo".into()),
            level: "warning".into(),
            message: "`todo` should not be in production code".into(),
            oip_category: None,
            confidence: 0.87,
            span: None,
            suggestion: None,
            timestamp: 1732752006,
            depyler_version: "3.21.0".into(),
        },
    ];

    // Convert to training examples
    let examples = convert_to_training_examples(&exports);

    // ═══════════════════════════════════════════════════════════════════════
    // df.head() style output
    // ═══════════════════════════════════════════════════════════════════════
    println!(
        "┌───────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ df.head()                                                              [{} rows] │",
        examples.len()
    );
    println!("├─────┬──────────────────────┬─────────────────────────┬────────┬─────────────────┤");
    println!("│ idx │ label                │ error_code/lint         │  conf  │ source          │");
    println!("├─────┼──────────────────────┼─────────────────────────┼────────┼─────────────────┤");

    for (i, ex) in examples.iter().enumerate() {
        let code = ex
            .error_code
            .as_deref()
            .or(ex.clippy_lint.as_deref())
            .unwrap_or("-");
        println!(
            "│ {:3} │ {:20} │ {:23} │ {:5.1}% │ {:15} │",
            i,
            ex.label.as_str(),
            &code[..code.len().min(23)],
            ex.confidence * 100.0,
            format!("{:?}", ex.source).replace("TrainingSource::", ""),
        );
    }
    println!("└─────┴──────────────────────┴─────────────────────────┴────────┴─────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════
    // df.describe() style output
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "┌───────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ df.describe()                                                                     │"
    );
    println!(
        "├───────────────────────────────────────────────────────────────────────────────────┤"
    );

    let conf_vals: Vec<f32> = examples.iter().map(|e| e.confidence).collect();
    let mean = conf_vals.iter().sum::<f32>() / conf_vals.len() as f32;
    let min = conf_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = conf_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sorted = conf_vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let variance =
        conf_vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / conf_vals.len() as f32;
    let std_dev = variance.sqrt();

    println!(
        "│                      confidence                                                   │"
    );
    println!(
        "│ count        {:6}                                                               │",
        examples.len()
    );
    println!(
        "│ mean         {:6.2}%                                                              │",
        mean * 100.0
    );
    println!(
        "│ std          {:6.2}%                                                              │",
        std_dev * 100.0
    );
    println!(
        "│ min          {:6.2}%                                                              │",
        min * 100.0
    );
    println!(
        "│ 50%          {:6.2}%                                                              │",
        median * 100.0
    );
    println!(
        "│ max          {:6.2}%                                                              │",
        max * 100.0
    );
    println!(
        "└───────────────────────────────────────────────────────────────────────────────────┘"
    );

    // ═══════════════════════════════════════════════════════════════════════
    // df['label'].value_counts() style output
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "┌───────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ df['label'].value_counts()                                                        │"
    );
    println!(
        "├───────────────────────────────────────────────────────────────────────────────────┤"
    );

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
    println!(
        "└───────────────────────────────────────────────────────────────────────────────────┘"
    );

    // ═══════════════════════════════════════════════════════════════════════
    // df.iloc[0] - Full record view
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "┌───────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ df.iloc[0]                                                                        │"
    );
    println!(
        "├───────────────────────────────────────────────────────────────────────────────────┤"
    );
    if let Some(ex) = examples.first() {
        println!(
            "│ message                {:60} │",
            &ex.message[..ex.message.len().min(60)]
        );
        println!("│ label                  {:60} │", ex.label.as_str());
        println!(
            "│ confidence             {:60} │",
            format!("{:.4}", ex.confidence)
        );
        println!(
            "│ error_code             {:60} │",
            ex.error_code.as_deref().unwrap_or("None")
        );
        println!(
            "│ clippy_lint            {:60} │",
            ex.clippy_lint.as_deref().unwrap_or("None")
        );
        println!(
            "│ has_suggestion         {:60} │",
            format!("{}", ex.has_suggestion)
        );
        println!(
            "│ suggestion_applicab.   {:60} │",
            format!("{:?}", ex.suggestion_applicability)
        );
        println!(
            "│ source                 {:60} │",
            format!("{:?}", ex.source)
        );
        println!(
            "│ timestamp              {:60} │",
            format!("{}", ex.timestamp)
        );
        println!("│ author                 {:60} │", &ex.author);
    }
    println!(
        "└───────────────────────────────────────────────────────────────────────────────────┘"
    );

    // ═══════════════════════════════════════════════════════════════════════
    // df.info() style output
    // ═══════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "┌───────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ df.info()                                                                         │"
    );
    println!(
        "├───────────────────────────────────────────────────────────────────────────────────┤"
    );
    println!(
        "│ <TrainingExample>                                                                 │"
    );
    println!(
        "│ RangeIndex: {} entries, 0 to {}                                                    │",
        examples.len(),
        examples.len() - 1
    );
    println!(
        "│ Data columns (total 12 columns):                                                  │"
    );
    println!(
        "│  #   Column                   Non-Null Count  Dtype                               │"
    );
    println!(
        "│ ---  ------                   --------------  -----                               │"
    );
    println!(
        "│  0   message                  {} non-null      String                              │",
        examples.len()
    );
    println!(
        "│  1   label                    {} non-null      DefectCategory                      │",
        examples.len()
    );
    println!(
        "│  2   confidence               {} non-null      f32                                 │",
        examples.len()
    );
    let ec_count = examples.iter().filter(|e| e.error_code.is_some()).count();
    let cl_count = examples.iter().filter(|e| e.clippy_lint.is_some()).count();
    println!(
        "│  3   error_code               {} non-null      Option<String>                      │",
        ec_count
    );
    println!(
        "│  4   clippy_lint              {} non-null      Option<String>                      │",
        cl_count
    );
    println!(
        "│  5   has_suggestion           {} non-null      bool                                │",
        examples.len()
    );
    println!(
        "│  6   suggestion_applicability {} non-null      Option<SuggestionApplicability>     │",
        examples
            .iter()
            .filter(|e| e.suggestion_applicability.is_some())
            .count()
    );
    println!(
        "│  7   source                   {} non-null      TrainingSource                      │",
        examples.len()
    );
    println!(
        "│ memory usage: ~{} bytes                                                           │",
        examples.len()
            * std::mem::size_of::<organizational_intelligence_plugin::training::TrainingExample>()
    );
    println!(
        "└───────────────────────────────────────────────────────────────────────────────────┘"
    );

    println!("\n✅ Data exploration complete!");
}

// Example: Import CITL (Compiler-in-the-Loop) training data from Depyler
// Demonstrates ground-truth label extraction from rustc/clippy diagnostics
//
// Usage:
//   cargo run --example citl_import
//
// This example shows NLP-014 CITL integration capabilities

use organizational_intelligence_plugin::citl::{
    clippy_to_defect_category, convert_to_training_examples, rustc_to_defect_category,
    CitlDataLoader, CitlLoaderConfig, DepylerExport, MergeStrategy,
};
use organizational_intelligence_plugin::classifier::DefectCategory;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🔧 CITL Integration - Depyler Ground-Truth Labels\n");
    println!("NLP-014: Compiler-in-the-Loop training data extraction\n");

    // Demonstrate rustc error code mappings
    println!("📋 rustc Error Code → DefectCategory Mappings:");
    let error_codes = [
        ("E0308", "Type mismatch"),
        ("E0382", "Use of moved value"),
        ("E0502", "Borrow while mutably borrowed"),
        ("E0277", "Trait bound not satisfied"),
        ("E0599", "Method not found"),
        ("E0425", "Unresolved name"),
    ];

    for (code, description) in &error_codes {
        if let Some(category) = rustc_to_defect_category(code) {
            println!("   {} ({}) → {}", code, description, category.as_str());
        }
    }

    // Demonstrate clippy lint mappings
    println!("\n📋 Clippy Lint → DefectCategory Mappings:");
    let clippy_lints = [
        ("clippy::unwrap_used", "Panicking unwrap"),
        ("clippy::expect_used", "Panicking expect"),
        ("clippy::todo", "Unfinished code"),
        ("clippy::cognitive_complexity", "Complex function"),
        ("clippy::needless_collect", "Unnecessary collect"),
    ];

    for (lint, description) in &clippy_lints {
        if let Some(category) = clippy_to_defect_category(lint) {
            println!("   {} ({}) → {}", lint, description, category.as_str());
        }
    }

    // Demonstrate DepylerExport parsing
    println!("\n📦 Parsing Depyler JSONL Export:");
    let sample_exports = vec![
        DepylerExport {
            source_file: "lib.rs".to_string(),
            error_code: Some("E0308".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "mismatched types: expected `i32`, found `&str`".to_string(),
            oip_category: None,
            confidence: 0.95,
            span: None,
            suggestion: None,
            timestamp: 1732752000,
            depyler_version: "3.21.0".to_string(),
        },
        DepylerExport {
            source_file: "main.rs".to_string(),
            error_code: None,
            clippy_lint: Some("clippy::unwrap_used".to_string()),
            level: "warning".to_string(),
            message: "used `unwrap()` on `Option` value".to_string(),
            oip_category: None,
            confidence: 0.88,
            span: None,
            suggestion: None,
            timestamp: 1732752001,
            depyler_version: "3.21.0".to_string(),
        },
        DepylerExport {
            source_file: "parser.rs".to_string(),
            error_code: Some("E0502".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "cannot borrow `self.data` as mutable".to_string(),
            oip_category: Some("OwnershipBorrow".to_string()),
            confidence: 0.92,
            span: None,
            suggestion: None,
            timestamp: 1732752002,
            depyler_version: "3.21.0".to_string(),
        },
    ];

    for export in &sample_exports {
        println!(
            "   {} [{}]: {}",
            export.source_file,
            export
                .error_code
                .as_deref()
                .unwrap_or(export.clippy_lint.as_deref().unwrap_or("unknown")),
            &export.message[..export.message.len().min(50)]
        );
    }

    // Convert to TrainingExamples
    println!("\n🔄 Converting to TrainingExamples:");
    let examples = convert_to_training_examples(&sample_exports);
    println!(
        "   Converted {} exports → {} training examples",
        sample_exports.len(),
        examples.len()
    );

    for example in &examples {
        println!(
            "   - [{}] {} (conf: {:.0}%)",
            example.label.as_str(),
            &example.message[..example.message.len().min(40)],
            example.confidence * 100.0
        );
    }

    // Demonstrate CitlDataLoader configuration
    println!("\n⚙️  CitlDataLoader Configuration:");
    let config = CitlLoaderConfig {
        batch_size: 1024,
        shuffle: true,
        min_confidence: 0.75,
        merge_strategy: MergeStrategy::Weighted(2),
        weight: 1.5,
    };

    let loader = CitlDataLoader::with_config(config);
    println!("   Batch size: {}", loader.config().batch_size);
    println!(
        "   Min confidence: {:.0}%",
        loader.config().min_confidence * 100.0
    );
    println!("   Shuffle: {}", loader.config().shuffle);
    println!("   Weight: {:.1}x", loader.config().weight);

    // Summary of categories covered
    println!("\n📚 CITL-Mapped Defect Categories:");
    let citl_categories = [
        (DefectCategory::TypeErrors, "E0308, E0412"),
        (DefectCategory::OwnershipBorrow, "E0502, E0503, E0505"),
        (DefectCategory::MemorySafety, "E0382, E0507"),
        (DefectCategory::TraitBounds, "E0277"),
        (DefectCategory::StdlibMapping, "E0425, E0433"),
        (DefectCategory::ASTTransform, "E0599, E0615"),
        (
            DefectCategory::ApiMisuse,
            "clippy::unwrap_used, clippy::expect_used",
        ),
        (
            DefectCategory::LogicErrors,
            "clippy::todo, clippy::unreachable",
        ),
        (
            DefectCategory::PerformanceIssues,
            "clippy::cognitive_complexity",
        ),
        (DefectCategory::IteratorChain, "clippy::needless_collect"),
    ];

    for (category, codes) in &citl_categories {
        println!("   {} ← {}", category.as_str(), codes);
    }

    println!("\n🎯 CITL Integration Complete!");
    println!("   Target: 5K+ ground-truth examples → 75%+ classifier accuracy");
    println!("   See: docs/specifications/nlp-014-citl-integration-spec.md");
}

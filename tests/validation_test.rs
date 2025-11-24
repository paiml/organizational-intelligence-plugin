/// Validation test for NLP-011: Validate ML classifier on depyler repository
use organizational_intelligence_plugin::classifier::{HybridClassifier, RuleBasedClassifier};
use organizational_intelligence_plugin::ml_trainer::MLTrainer;
use std::path::Path;

#[test]
#[ignore] // Run with: cargo test --test validation_test -- --ignored --nocapture
fn depyler_model_validation() {
    println!("\n📊 Depyler ML Model Validation Report");
    println!("=====================================\n");

    // Load the trained model (skip if model doesn't exist)
    let model_path = Path::new("/tmp/depyler-model.bin");
    if !model_path.exists() {
        println!("⚠️  Model not found at /tmp/depyler-model.bin");
        println!("   Run: cargo run --release -- extract-training-data --repo ../depyler --output /tmp/depyler-training.json");
        println!("   Then: cargo run --release -- train-classifier --input /tmp/depyler-training.json --output /tmp/depyler-model.bin");
        return;
    }

    println!("📂 Loading ML model from /tmp/depyler-model.bin...");
    let ml_model = MLTrainer::load_model(model_path).expect("Failed to load model");
    println!("   ✅ Model loaded successfully");
    println!(
        "   Training accuracy: {:.2}%",
        ml_model.metadata.train_accuracy * 100.0
    );
    println!(
        "   Validation accuracy: {:.2}%",
        ml_model.metadata.validation_accuracy * 100.0
    );
    if let Some(test_acc) = ml_model.metadata.test_accuracy {
        println!("   Test accuracy: {:.2}%", test_acc * 100.0);
    }
    println!("   Classes: {}\n", ml_model.metadata.n_classes);

    // Create classifiers
    let ml_classifier = HybridClassifier::new_hybrid(ml_model, 0.60);
    let rule_based_classifier = RuleBasedClassifier::new();

    // Test cases from depyler repository (transpiler-specific)
    let test_cases = vec![
        // ASTTransform
        (
            "fix: correct AST node transformation for match expressions",
            "ASTTransform",
            true,
        ),
        (
            "fix: handle nested function definitions in AST",
            "ASTTransform",
            true,
        ),
        (
            "fix: AST visitor pattern for lambda expressions",
            "ASTTransform",
            true,
        ),
        // OwnershipBorrow
        (
            "fix: resolve borrow checker error in iterator",
            "OwnershipBorrow",
            true,
        ),
        (
            "fix: lifetime annotation for returned reference",
            "OwnershipBorrow",
            true,
        ),
        // StdlibMapping
        (
            "fix: map Python os.path to std::path correctly",
            "StdlibMapping",
            true,
        ),
        // TypeAnnotationGaps
        (
            "fix: add missing type annotation for closure",
            "TypeAnnotationGaps",
            true,
        ),
        // General defects
        (
            "fix: null pointer dereference in parser",
            "MemorySafety",
            true,
        ),
        (
            "fix: race condition in parallel compilation",
            "ConcurrencyBugs",
            true,
        ),
        // Non-defects
        ("feat: add new optimization pass", "N/A", false),
        ("docs: update README", "N/A", false),
    ];

    println!("🔍 Testing on {} sample messages\n", test_cases.len());
    println!(
        "{:<65} {:<20} {:<20} {:<10}",
        "Message", "ML Result", "RB Result", "Expected"
    );
    println!("{}", "-".repeat(115));

    let mut ml_correct = 0;
    let mut rb_correct = 0;
    let mut ml_detected = 0;
    let mut rb_detected = 0;

    for (message, expected_category, is_defect) in &test_cases {
        let ml_result = ml_classifier.classify_from_message(message);
        let rb_result = rule_based_classifier.classify_from_message(message);

        let ml_category = ml_result
            .as_ref()
            .map(|c| format!("{:?}", c.category))
            .unwrap_or("None".to_string());
        let rb_category = rb_result
            .as_ref()
            .map(|c| format!("{:?}", c.category))
            .unwrap_or("None".to_string());

        let display_msg = if message.len() > 62 {
            &message[..62]
        } else {
            message
        };
        println!(
            "{:<65} {:<20} {:<20} {:<10}",
            display_msg,
            &ml_category[..ml_category.len().min(20)],
            &rb_category[..rb_category.len().min(20)],
            expected_category
        );

        // Track detection accuracy
        if *is_defect {
            if ml_result.is_some() {
                ml_detected += 1;
                // Check if category matches
                if let Some(ref result) = ml_result {
                    if format!("{:?}", result.category) == *expected_category {
                        ml_correct += 1;
                    }
                }
            }
            if rb_result.is_some() {
                rb_detected += 1;
                if let Some(ref result) = rb_result {
                    if format!("{:?}", result.category) == *expected_category {
                        rb_correct += 1;
                    }
                }
            }
        }
    }

    let defect_count = test_cases
        .iter()
        .filter(|(_, _, is_defect)| *is_defect)
        .count();

    println!("\n📈 Performance Summary:");
    println!("   Total defect messages: {}", defect_count);
    println!("\n   ML Classifier:");
    println!(
        "      Detected: {} ({:.1}%)",
        ml_detected,
        (ml_detected as f64 / defect_count as f64) * 100.0
    );
    println!(
        "      Correct category: {} ({:.1}%)",
        ml_correct,
        (ml_correct as f64 / defect_count as f64) * 100.0
    );
    println!("\n   Rule-Based Classifier:");
    println!(
        "      Detected: {} ({:.1}%)",
        rb_detected,
        (rb_detected as f64 / defect_count as f64) * 100.0
    );
    println!(
        "      Correct category: {} ({:.1}%)",
        rb_correct,
        (rb_correct as f64 / defect_count as f64) * 100.0
    );

    println!("\n💡 Insights:");
    if ml_detected > rb_detected {
        println!(
            "   ✅ ML classifier detected more defects than rule-based (+{})",
            ml_detected - rb_detected
        );
    } else if rb_detected > ml_detected {
        println!(
            "   ⚠️  Rule-based detected more defects than ML (+{})",
            rb_detected - ml_detected
        );
    } else {
        println!("   ≈ ML and rule-based detected same number of defects");
    }

    println!("\n🎯 Validation Complete!");
}

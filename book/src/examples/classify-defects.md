# Example: Classify Defects

This example demonstrates the rule-based classifier (Phase 1) for defect categorization from commit messages.

## Running the Example

```bash
cargo run --example classify_defects
```

## What It Does

1. **Initializes the Rule-Based Classifier** - Creates a `RuleBasedClassifier` with predefined pattern matchers for 10 defect categories
2. **Classifies Sample Commit Messages** - Tests against various bug-fix commit messages
3. **Shows Classification Results** - Displays category, confidence score, explanation, and matched patterns
4. **Lists All Categories** - Demonstrates the full 10-category taxonomy

## Sample Output

```
🤖 Rule-Based Defect Classifier - Example

Phase 1: Heuristic pattern matching with confidence scores

📋 Classifying 11 commit messages:

Message: "fix: use-after-free in buffer handling"
  ✅ Category: Memory Safety
  📊 Confidence: 95%
  💡 Explanation: Matches memory safety patterns: use-after-free
  🔍 Matched patterns: use-after-free

Message: "fix: race condition in async handler"
  ✅ Category: Concurrency Bugs
  📊 Confidence: 90%
  💡 Explanation: Matches concurrency patterns: race condition
  🔍 Matched patterns: race condition

Message: "docs: update installation guide"
  ⏭️  Not classified as a defect (documentation/feature/etc)

📊 Summary:
   Classified: 9 (82%)
   Unclassified: 2 (18%)

📚 All 10 Defect Categories:
   1. Memory Safety
   2. Concurrency Bugs
   3. Logic Errors
   4. API Misuse
   5. Resource Leaks
   6. Type Errors
   7. Configuration Errors
   8. Security Vulnerabilities
   9. Performance Issues
   10. Integration Failures
```

## Key Concepts

- **Confidence Scoring**: Each classification includes a confidence percentage based on pattern match strength
- **Pattern Matching**: Uses regex-based heuristics to detect defect indicators in commit messages
- **Multi-Category Support**: Handles 10 distinct defect categories
- **Non-Defect Filtering**: Correctly ignores documentation, feature, and refactoring commits

## API Usage

```rust
use organizational_intelligence_plugin::classifier::RuleBasedClassifier;

let classifier = RuleBasedClassifier::new();
let message = "fix: null pointer dereference in parser";

if let Some(classification) = classifier.classify_from_message(message) {
    println!("Category: {}", classification.category.as_str());
    println!("Confidence: {:.0}%", classification.confidence * 100.0);
}
```

## See Also

- [Three-Tier Classification](../architecture/confidence-routing.md)
- [18-Category Defect Taxonomy](../core-concepts/defect-taxonomy.md)

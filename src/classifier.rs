// Rule-based defect classifier
// Phase 1: Heuristic-based classification with confidence scores and explanations
// Toyota Way: Start simple, collect data for Phase 2 ML

use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::debug;

/// Defect categories based on research literature
/// See specification Section 2.2.3 and Section 5.2 (Expanded Taxonomy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
    // General defect categories (10)
    MemorySafety,
    ConcurrencyBugs,
    LogicErrors,
    ApiMisuse,
    ResourceLeaks,
    TypeErrors,
    ConfigurationErrors,
    SecurityVulnerabilities,
    PerformanceIssues,
    IntegrationFailures,
    // Transpiler-specific categories (8)
    OperatorPrecedence,
    TypeAnnotationGaps,
    StdlibMapping,
    ASTTransform,
    ComprehensionBugs,
    IteratorChain,
    OwnershipBorrow,
    TraitBounds,
}

impl DefectCategory {
    /// Get human-readable name for the category
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MemorySafety => "Memory Safety",
            Self::ConcurrencyBugs => "Concurrency Bugs",
            Self::LogicErrors => "Logic Errors",
            Self::ApiMisuse => "API Misuse",
            Self::ResourceLeaks => "Resource Leaks",
            Self::TypeErrors => "Type Errors",
            Self::ConfigurationErrors => "Configuration Errors",
            Self::SecurityVulnerabilities => "Security Vulnerabilities",
            Self::PerformanceIssues => "Performance Issues",
            Self::IntegrationFailures => "Integration Failures",
            Self::OperatorPrecedence => "Operator Precedence",
            Self::TypeAnnotationGaps => "Type Annotation Gaps",
            Self::StdlibMapping => "Stdlib Mapping",
            Self::ASTTransform => "AST Transform",
            Self::ComprehensionBugs => "Comprehension Bugs",
            Self::IteratorChain => "Iterator Chain",
            Self::OwnershipBorrow => "Ownership/Borrow",
            Self::TraitBounds => "Trait Bounds",
        }
    }
}

impl fmt::Display for DefectCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the enum variant name for serialization compatibility
        match self {
            Self::MemorySafety => write!(f, "MemorySafety"),
            Self::ConcurrencyBugs => write!(f, "ConcurrencyBugs"),
            Self::LogicErrors => write!(f, "LogicErrors"),
            Self::ApiMisuse => write!(f, "ApiMisuse"),
            Self::ResourceLeaks => write!(f, "ResourceLeaks"),
            Self::TypeErrors => write!(f, "TypeErrors"),
            Self::ConfigurationErrors => write!(f, "ConfigurationErrors"),
            Self::SecurityVulnerabilities => write!(f, "SecurityVulnerabilities"),
            Self::PerformanceIssues => write!(f, "PerformanceIssues"),
            Self::IntegrationFailures => write!(f, "IntegrationFailures"),
            Self::OperatorPrecedence => write!(f, "OperatorPrecedence"),
            Self::TypeAnnotationGaps => write!(f, "TypeAnnotationGaps"),
            Self::StdlibMapping => write!(f, "StdlibMapping"),
            Self::ASTTransform => write!(f, "ASTTransform"),
            Self::ComprehensionBugs => write!(f, "ComprehensionBugs"),
            Self::IteratorChain => write!(f, "IteratorChain"),
            Self::OwnershipBorrow => write!(f, "OwnershipBorrow"),
            Self::TraitBounds => write!(f, "TraitBounds"),
        }
    }
}

/// Classification result with confidence and explanation
/// Following Toyota Way: Respect for People - provide explanations for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    pub category: DefectCategory,
    pub confidence: f32, // 0.0 to 1.0
    pub explanation: String,
    pub matched_patterns: Vec<String>,
}

/// Pattern matching rule
#[derive(Debug, Clone)]
struct Rule {
    category: DefectCategory,
    patterns: Vec<&'static str>,
    confidence: f32,
}

/// Rule-based classifier
/// Phase 1: Simple pattern matching on commit messages
/// Phase 2: Will evolve to ML-based with user feedback
pub struct RuleBasedClassifier {
    rules: Vec<Rule>,
}

impl RuleBasedClassifier {
    /// Create a new rule-based classifier with predefined patterns
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::classifier::RuleBasedClassifier;
    ///
    /// let classifier = RuleBasedClassifier::new();
    /// ```
    pub fn new() -> Self {
        let rules = vec![
            // Memory Safety patterns
            Rule {
                category: DefectCategory::MemorySafety,
                patterns: vec![
                    "use after free",
                    "use-after-free",
                    "null pointer",
                    "nullptr",
                    "buffer overflow",
                    "memory leak",
                    "dangling pointer",
                    "double free",
                    "heap corruption",
                ],
                confidence: 0.85,
            },
            // Concurrency patterns
            Rule {
                category: DefectCategory::ConcurrencyBugs,
                patterns: vec![
                    "race condition",
                    "data race",
                    "deadlock",
                    "atomicity",
                    "thread safety",
                    "concurrent",
                    "synchronization",
                    "mutex",
                    "lock contention",
                ],
                confidence: 0.80,
            },
            // Security patterns
            Rule {
                category: DefectCategory::SecurityVulnerabilities,
                patterns: vec![
                    "sql injection",
                    "xss",
                    "cross-site scripting",
                    "authentication",
                    "authorization",
                    "security",
                    "vulnerability",
                    "exploit",
                    "cve-",
                ],
                confidence: 0.90,
            },
            // Logic Error patterns
            Rule {
                category: DefectCategory::LogicErrors,
                patterns: vec![
                    "off by one",
                    "off-by-one",
                    "boundary",
                    "incorrect logic",
                    "wrong condition",
                    "infinite loop",
                ],
                confidence: 0.70,
            },
            // API Misuse patterns
            Rule {
                category: DefectCategory::ApiMisuse,
                patterns: vec![
                    "api misuse",
                    "wrong parameter",
                    "incorrect usage",
                    "missing error handling",
                    "unchecked error",
                ],
                confidence: 0.75,
            },
            // Resource Leak patterns
            Rule {
                category: DefectCategory::ResourceLeaks,
                patterns: vec![
                    "resource leak",
                    "file handle leak",
                    "connection leak",
                    "not closed",
                    "forgot to close",
                ],
                confidence: 0.80,
            },
            // Type Error patterns
            Rule {
                category: DefectCategory::TypeErrors,
                patterns: vec![
                    "type error",
                    "type mismatch",
                    "casting error",
                    "serialization",
                    "deserialization",
                ],
                confidence: 0.75,
            },
            // Configuration patterns
            Rule {
                category: DefectCategory::ConfigurationErrors,
                patterns: vec![
                    "configuration",
                    "config",
                    "environment variable",
                    "missing env",
                    "settings",
                ],
                confidence: 0.70,
            },
            // Performance patterns
            Rule {
                category: DefectCategory::PerformanceIssues,
                patterns: vec![
                    "performance",
                    "slow",
                    "inefficient",
                    "n+1 query",
                    "optimization",
                ],
                confidence: 0.65,
            },
            // Integration patterns
            Rule {
                category: DefectCategory::IntegrationFailures,
                patterns: vec![
                    "integration",
                    "compatibility",
                    "version mismatch",
                    "breaking change",
                    "api change",
                ],
                confidence: 0.70,
            },
            // Transpiler-specific patterns
            // Operator Precedence patterns
            Rule {
                category: DefectCategory::OperatorPrecedence,
                patterns: vec![
                    "operator precedence",
                    "parentheses",
                    "parse expression",
                    "order of operations",
                    "precedence",
                    "expression parsing",
                    "operator order",
                ],
                confidence: 0.80,
            },
            // Type Annotation Gaps patterns
            Rule {
                category: DefectCategory::TypeAnnotationGaps,
                patterns: vec![
                    "type annotation",
                    "type hint",
                    "unsupported type",
                    "generic type",
                    "type parameter",
                    "annotation",
                    "typing",
                ],
                confidence: 0.75,
            },
            // Stdlib Mapping patterns
            Rule {
                category: DefectCategory::StdlibMapping,
                patterns: vec![
                    "stdlib",
                    "standard library",
                    "python to rust",
                    "library mapping",
                    "std::",
                    "builtin",
                    "library conversion",
                ],
                confidence: 0.80,
            },
            // AST Transform patterns
            Rule {
                category: DefectCategory::ASTTransform,
                patterns: vec![
                    "ast",
                    "hir",
                    "codegen",
                    "transform",
                    "syntax tree",
                    "ast node",
                    "tree traversal",
                ],
                confidence: 0.85,
            },
            // Comprehension Bugs patterns
            Rule {
                category: DefectCategory::ComprehensionBugs,
                patterns: vec![
                    "comprehension",
                    "list comprehension",
                    "dict comprehension",
                    "set comprehension",
                    "generator",
                    "generator expression",
                ],
                confidence: 0.80,
            },
            // Iterator Chain patterns
            Rule {
                category: DefectCategory::IteratorChain,
                patterns: vec![
                    "iterator",
                    "into_iter",
                    ".map(",
                    ".filter(",
                    ".chain(",
                    "iterator chain",
                    "iter method",
                ],
                confidence: 0.80,
            },
            // Ownership/Borrow patterns
            Rule {
                category: DefectCategory::OwnershipBorrow,
                patterns: vec![
                    "ownership",
                    "borrow",
                    "lifetime",
                    "borrow checker",
                    "move",
                    "borrowed value",
                    "lifetime parameter",
                ],
                confidence: 0.85,
            },
            // Trait Bounds patterns
            Rule {
                category: DefectCategory::TraitBounds,
                patterns: vec![
                    "trait bound",
                    "generic constraint",
                    "where clause",
                    "impl trait",
                    "trait constraint",
                    "bound",
                ],
                confidence: 0.80,
            },
        ];

        Self { rules }
    }

    /// Classify a defect based on commit message
    ///
    /// # Arguments
    /// * `message` - Commit message text
    ///
    /// # Returns
    /// * `Some(Classification)` if patterns match
    /// * `None` if no patterns match (not a defect fix)
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::classifier::RuleBasedClassifier;
    ///
    /// let classifier = RuleBasedClassifier::new();
    /// let result = classifier.classify_from_message("fix: null pointer dereference");
    ///
    /// assert!(result.is_some());
    /// ```
    pub fn classify_from_message(&self, message: &str) -> Option<Classification> {
        let message_lower = message.to_lowercase();

        debug!("Classifying message: {}", message);

        let mut matches: Vec<(DefectCategory, f32, Vec<String>)> = Vec::new();

        // Check each rule
        for rule in &self.rules {
            let mut matched_patterns = Vec::new();

            for pattern in &rule.patterns {
                if message_lower.contains(pattern) {
                    matched_patterns.push(pattern.to_string());
                }
            }

            if !matched_patterns.is_empty() {
                // Boost confidence if multiple patterns match
                let confidence_boost = (matched_patterns.len() - 1) as f32 * 0.05;
                let adjusted_confidence = (rule.confidence + confidence_boost).min(0.95);

                matches.push((rule.category, adjusted_confidence, matched_patterns));
            }
        }

        if matches.is_empty() {
            debug!("No patterns matched for message");
            return None;
        }

        // Sort by confidence (highest first)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take the highest confidence match
        let (category, confidence, matched_patterns) = matches.into_iter().next().unwrap();

        let explanation = format!(
            "Classified as '{}' based on patterns: {}. Confidence: {:.0}%",
            category.as_str(),
            matched_patterns.join(", "),
            confidence * 100.0
        );

        debug!(
            "Classification: {:?} with confidence {}",
            category, confidence
        );

        Some(Classification {
            category,
            confidence,
            explanation,
            matched_patterns,
        })
    }
}

impl Default for RuleBasedClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let _classifier = RuleBasedClassifier::new();
    }

    #[test]
    fn test_all_categories_covered() {
        let classifier = RuleBasedClassifier::new();

        // Verify we have rules for all 18 categories (10 general + 8 transpiler)
        let mut categories_covered = std::collections::HashSet::new();
        for rule in &classifier.rules {
            categories_covered.insert(rule.category);
        }

        assert_eq!(
            categories_covered.len(),
            18,
            "Should have rules for all 18 categories (10 general + 8 transpiler)"
        );
    }

    #[test]
    fn test_pattern_matching() {
        let classifier = RuleBasedClassifier::new();

        let test_cases = vec![
            ("fix: use-after-free bug", DefectCategory::MemorySafety),
            ("fix: race condition", DefectCategory::ConcurrencyBugs),
            (
                "security: prevent SQL injection",
                DefectCategory::SecurityVulnerabilities,
            ),
        ];

        for (message, expected_category) in test_cases {
            let result = classifier.classify_from_message(message);
            assert!(result.is_some(), "Should classify: {}", message);
            assert_eq!(result.unwrap().category, expected_category);
        }
    }

    #[test]
    fn test_non_defect_returns_none() {
        let classifier = RuleBasedClassifier::new();

        let non_defect_messages = vec![
            "docs: update README",
            "chore: bump version",
            "feat: add new feature",
            "refactor: simplify code",
        ];

        for message in non_defect_messages {
            let result = classifier.classify_from_message(message);
            assert!(
                result.is_none(),
                "Should not classify as defect: {}",
                message
            );
        }
    }

    #[test]
    fn test_defect_category_as_str() {
        // General categories
        assert_eq!(DefectCategory::MemorySafety.as_str(), "Memory Safety");
        assert_eq!(DefectCategory::ConcurrencyBugs.as_str(), "Concurrency Bugs");
        assert_eq!(DefectCategory::LogicErrors.as_str(), "Logic Errors");
        assert_eq!(DefectCategory::ApiMisuse.as_str(), "API Misuse");
        assert_eq!(DefectCategory::ResourceLeaks.as_str(), "Resource Leaks");
        assert_eq!(DefectCategory::TypeErrors.as_str(), "Type Errors");
        assert_eq!(
            DefectCategory::ConfigurationErrors.as_str(),
            "Configuration Errors"
        );
        assert_eq!(
            DefectCategory::SecurityVulnerabilities.as_str(),
            "Security Vulnerabilities"
        );
        assert_eq!(
            DefectCategory::PerformanceIssues.as_str(),
            "Performance Issues"
        );
        assert_eq!(
            DefectCategory::IntegrationFailures.as_str(),
            "Integration Failures"
        );
        // Transpiler categories
        assert_eq!(
            DefectCategory::OperatorPrecedence.as_str(),
            "Operator Precedence"
        );
        assert_eq!(
            DefectCategory::TypeAnnotationGaps.as_str(),
            "Type Annotation Gaps"
        );
        assert_eq!(DefectCategory::StdlibMapping.as_str(), "Stdlib Mapping");
        assert_eq!(DefectCategory::ASTTransform.as_str(), "AST Transform");
        assert_eq!(
            DefectCategory::ComprehensionBugs.as_str(),
            "Comprehension Bugs"
        );
        assert_eq!(DefectCategory::IteratorChain.as_str(), "Iterator Chain");
        assert_eq!(DefectCategory::OwnershipBorrow.as_str(), "Ownership/Borrow");
        assert_eq!(DefectCategory::TraitBounds.as_str(), "Trait Bounds");
    }

    #[test]
    fn test_defect_category_display() {
        // General categories
        assert_eq!(format!("{}", DefectCategory::MemorySafety), "MemorySafety");
        assert_eq!(
            format!("{}", DefectCategory::ConcurrencyBugs),
            "ConcurrencyBugs"
        );
        assert_eq!(format!("{}", DefectCategory::LogicErrors), "LogicErrors");
        assert_eq!(format!("{}", DefectCategory::ApiMisuse), "ApiMisuse");
        assert_eq!(
            format!("{}", DefectCategory::ResourceLeaks),
            "ResourceLeaks"
        );
        assert_eq!(format!("{}", DefectCategory::TypeErrors), "TypeErrors");
        assert_eq!(
            format!("{}", DefectCategory::ConfigurationErrors),
            "ConfigurationErrors"
        );
        assert_eq!(
            format!("{}", DefectCategory::SecurityVulnerabilities),
            "SecurityVulnerabilities"
        );
        assert_eq!(
            format!("{}", DefectCategory::PerformanceIssues),
            "PerformanceIssues"
        );
        assert_eq!(
            format!("{}", DefectCategory::IntegrationFailures),
            "IntegrationFailures"
        );
        // Transpiler categories
        assert_eq!(
            format!("{}", DefectCategory::OperatorPrecedence),
            "OperatorPrecedence"
        );
        assert_eq!(
            format!("{}", DefectCategory::TypeAnnotationGaps),
            "TypeAnnotationGaps"
        );
        assert_eq!(
            format!("{}", DefectCategory::StdlibMapping),
            "StdlibMapping"
        );
        assert_eq!(format!("{}", DefectCategory::ASTTransform), "ASTTransform");
        assert_eq!(
            format!("{}", DefectCategory::ComprehensionBugs),
            "ComprehensionBugs"
        );
        assert_eq!(
            format!("{}", DefectCategory::IteratorChain),
            "IteratorChain"
        );
        assert_eq!(
            format!("{}", DefectCategory::OwnershipBorrow),
            "OwnershipBorrow"
        );
        assert_eq!(format!("{}", DefectCategory::TraitBounds), "TraitBounds");
    }

    #[test]
    fn test_default_constructor() {
        let classifier = RuleBasedClassifier::default();
        assert_eq!(classifier.rules.len(), 18);
    }

    #[test]
    fn test_empty_message() {
        let classifier = RuleBasedClassifier::new();
        let result = classifier.classify_from_message("");
        assert!(result.is_none());
    }

    #[test]
    fn test_case_insensitive_matching() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier.classify_from_message("Fix: NULL POINTER dereference");
        assert!(result.is_some());
        assert_eq!(result.unwrap().category, DefectCategory::MemorySafety);
    }

    #[test]
    fn test_multiple_patterns_boost_confidence() {
        let classifier = RuleBasedClassifier::new();

        // Message with multiple memory safety patterns
        let result = classifier
            .classify_from_message("fix: null pointer and buffer overflow")
            .unwrap();

        assert_eq!(result.category, DefectCategory::MemorySafety);
        // Base confidence 0.85 + 0.05 boost for 2nd pattern = 0.90
        assert!(result.confidence >= 0.85);
        assert_eq!(result.matched_patterns.len(), 2);
    }

    #[test]
    fn test_confidence_capped_at_95_percent() {
        let classifier = RuleBasedClassifier::new();

        // Message with many security patterns to exceed 0.95 cap
        let result = classifier
            .classify_from_message(
                "security vulnerability exploit with sql injection and xss and cve-2024-1234",
            )
            .unwrap();

        assert_eq!(result.category, DefectCategory::SecurityVulnerabilities);
        assert!(result.confidence <= 0.95);
    }

    #[test]
    fn test_highest_confidence_wins() {
        let classifier = RuleBasedClassifier::new();

        // "security" has higher confidence (0.90) than "performance" (0.65)
        let result = classifier
            .classify_from_message("fix security and performance issues")
            .unwrap();

        assert_eq!(result.category, DefectCategory::SecurityVulnerabilities);
    }

    #[test]
    fn test_all_categories_classifiable() {
        let classifier = RuleBasedClassifier::new();

        let test_cases = vec![
            // General categories
            ("null pointer bug", DefectCategory::MemorySafety),
            ("race condition fix", DefectCategory::ConcurrencyBugs),
            ("off by one error", DefectCategory::LogicErrors),
            ("api misuse fix", DefectCategory::ApiMisuse),
            ("resource leak fix", DefectCategory::ResourceLeaks),
            ("type error fix", DefectCategory::TypeErrors),
            ("configuration bug", DefectCategory::ConfigurationErrors),
            ("security fix", DefectCategory::SecurityVulnerabilities),
            ("performance fix", DefectCategory::PerformanceIssues),
            ("integration failure", DefectCategory::IntegrationFailures),
            // Transpiler categories
            (
                "fix operator precedence issue",
                DefectCategory::OperatorPrecedence,
            ),
            (
                "type annotation not supported",
                DefectCategory::TypeAnnotationGaps,
            ),
            ("stdlib mapping bug", DefectCategory::StdlibMapping),
            ("ast transform error", DefectCategory::ASTTransform),
            ("list comprehension bug", DefectCategory::ComprehensionBugs),
            ("iterator chain issue", DefectCategory::IteratorChain),
            ("ownership error", DefectCategory::OwnershipBorrow),
            ("trait bound issue", DefectCategory::TraitBounds),
        ];

        for (message, expected_category) in test_cases {
            let result = classifier.classify_from_message(message);
            assert!(result.is_some(), "Should classify: {}", message);
            assert_eq!(
                result.unwrap().category,
                expected_category,
                "Failed for: {}",
                message
            );
        }
    }

    #[test]
    fn test_classification_struct_fields() {
        let classifier = RuleBasedClassifier::new();
        let result = classifier
            .classify_from_message("fix: deadlock in mutex")
            .unwrap();

        assert_eq!(result.category, DefectCategory::ConcurrencyBugs);
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert!(!result.explanation.is_empty());
        assert!(!result.matched_patterns.is_empty());
    }

    #[test]
    fn test_explanation_format() {
        let classifier = RuleBasedClassifier::new();
        let result = classifier
            .classify_from_message("fix: sql injection vulnerability")
            .unwrap();

        assert!(result.explanation.contains("Security Vulnerabilities"));
        assert!(result.explanation.contains("sql injection"));
        assert!(result.explanation.contains("Confidence:"));
        assert!(result.explanation.contains("%"));
    }

    #[test]
    fn test_matched_patterns_populated() {
        let classifier = RuleBasedClassifier::new();
        let result = classifier
            .classify_from_message("fix: double free and memory leak")
            .unwrap();

        assert_eq!(result.matched_patterns.len(), 2);
        assert!(result.matched_patterns.contains(&"double free".to_string()));
        assert!(result.matched_patterns.contains(&"memory leak".to_string()));
    }

    #[test]
    fn test_transpiler_operator_precedence_classification() {
        let classifier = RuleBasedClassifier::new();

        let test_cases = vec![
            "fix: operator precedence bug in expression parser",
            "fix: incorrect parentheses handling",
            "fix: parse expression order of operations",
        ];

        for message in test_cases {
            let result = classifier.classify_from_message(message);
            assert!(result.is_some(), "Should classify: {}", message);
            assert_eq!(
                result.unwrap().category,
                DefectCategory::OperatorPrecedence,
                "Failed for: {}",
                message
            );
        }
    }

    #[test]
    fn test_transpiler_type_annotation_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: type annotation gap in generic type")
            .unwrap();

        assert_eq!(result.category, DefectCategory::TypeAnnotationGaps);
        assert!(result.matched_patterns.len() >= 2);
    }

    #[test]
    fn test_transpiler_ownership_classification() {
        let classifier = RuleBasedClassifier::new();

        let test_cases = vec![
            "fix: borrow checker error in iterator",
            "fix: lifetime parameter issue",
            "fix: ownership move bug",
        ];

        for message in test_cases {
            let result = classifier.classify_from_message(message);
            assert!(result.is_some(), "Should classify: {}", message);
            assert_eq!(
                result.unwrap().category,
                DefectCategory::OwnershipBorrow,
                "Failed for: {}",
                message
            );
        }
    }

    #[test]
    fn test_transpiler_comprehension_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: dict comprehension generation bug")
            .unwrap();

        assert_eq!(result.category, DefectCategory::ComprehensionBugs);
        assert!(result.confidence >= 0.80);
    }

    #[test]
    fn test_transpiler_iterator_chain_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: .map( and .filter( iterator chain issue")
            .unwrap();

        assert_eq!(result.category, DefectCategory::IteratorChain);
        assert!(result.matched_patterns.len() >= 2);
    }

    #[test]
    fn test_transpiler_ast_transform_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: ast node transform in codegen")
            .unwrap();

        assert_eq!(result.category, DefectCategory::ASTTransform);
        assert!(result.confidence >= 0.85);
    }

    #[test]
    fn test_transpiler_stdlib_mapping_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: stdlib mapping from python to rust")
            .unwrap();

        assert_eq!(result.category, DefectCategory::StdlibMapping);
    }

    #[test]
    fn test_transpiler_trait_bounds_classification() {
        let classifier = RuleBasedClassifier::new();

        let result = classifier
            .classify_from_message("fix: trait bound issue in where clause")
            .unwrap();

        assert_eq!(result.category, DefectCategory::TraitBounds);
        assert!(result.matched_patterns.len() >= 2);
    }
}

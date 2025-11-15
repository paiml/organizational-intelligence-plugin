// Rule-based defect classifier
// Phase 1: Heuristic-based classification with confidence scores and explanations
// Toyota Way: Start simple, collect data for Phase 2 ML

use serde::{Deserialize, Serialize};
use tracing::debug;

/// Defect categories based on research literature
/// See specification Section 2.2.3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
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

        // Verify we have rules for all 10 categories
        let mut categories_covered = std::collections::HashSet::new();
        for rule in &classifier.rules {
            categories_covered.insert(rule.category);
        }

        assert_eq!(
            categories_covered.len(),
            10,
            "Should have rules for all 10 categories"
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
}

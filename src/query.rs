//! Natural Language Query Parser
//!
//! Implements Section 10: Query System & Natural Language Interface
//! Phase 1: Basic pattern matching for common queries

use anyhow::Result;

/// Query type extracted from natural language
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// "show me most common defect"
    MostCommonDefect,
    /// "count defects by category"
    CountByCategory,
    /// "show all defects"
    ListAll,
    /// Unknown query
    Unknown(String),
}

/// Parsed query with parameters
#[derive(Debug, Clone)]
pub struct Query {
    pub query_type: QueryType,
    pub original: String,
}

/// Natural language query parser
pub struct QueryParser;

impl QueryParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse natural language query into structured query
    pub fn parse(&self, input: &str) -> Result<Query> {
        let lower = input.to_lowercase();

        let query_type = if lower.contains("most common") || lower.contains("top defect") {
            QueryType::MostCommonDefect
        } else if lower.contains("count")
            && (lower.contains("category") || lower.contains("defect"))
        {
            QueryType::CountByCategory
        } else if lower.contains("show") && lower.contains("all") {
            QueryType::ListAll
        } else {
            QueryType::Unknown(input.to_string())
        };

        Ok(Query {
            query_type,
            original: input.to_string(),
        })
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = QueryParser::new();
    }

    #[test]
    fn test_most_common_defect() {
        let parser = QueryParser::new();

        let q1 = parser.parse("show me most common defect").unwrap();
        assert_eq!(q1.query_type, QueryType::MostCommonDefect);

        let q2 = parser.parse("what is the top defect?").unwrap();
        assert_eq!(q2.query_type, QueryType::MostCommonDefect);
    }

    #[test]
    fn test_count_by_category() {
        let parser = QueryParser::new();

        let q = parser.parse("count defects by category").unwrap();
        assert_eq!(q.query_type, QueryType::CountByCategory);
    }

    #[test]
    fn test_list_all() {
        let parser = QueryParser::new();

        let q = parser.parse("show all defects").unwrap();
        assert_eq!(q.query_type, QueryType::ListAll);
    }

    #[test]
    fn test_unknown_query() {
        let parser = QueryParser::new();

        let q = parser.parse("hello world").unwrap();
        assert!(matches!(q.query_type, QueryType::Unknown(_)));
    }

    #[test]
    fn test_parser_default() {
        let parser = QueryParser;
        let q = parser.parse("show me most common defect").unwrap();
        assert_eq!(q.query_type, QueryType::MostCommonDefect);
    }

    #[test]
    fn test_query_original_preserved() {
        let parser = QueryParser::new();
        let input = "Count Defects BY Category";
        let q = parser.parse(input).unwrap();
        assert_eq!(q.original, input);
        assert_eq!(q.query_type, QueryType::CountByCategory);
    }

    #[test]
    fn test_count_with_category_keyword() {
        let parser = QueryParser::new();
        let q = parser.parse("count by category").unwrap();
        assert_eq!(q.query_type, QueryType::CountByCategory);
    }

    #[test]
    fn test_count_with_defect_keyword() {
        let parser = QueryParser::new();
        let q = parser.parse("count defect types").unwrap();
        assert_eq!(q.query_type, QueryType::CountByCategory);
    }

    #[test]
    fn test_case_insensitive_parsing() {
        let parser = QueryParser::new();

        let q1 = parser.parse("SHOW ALL DEFECTS").unwrap();
        assert_eq!(q1.query_type, QueryType::ListAll);

        let q2 = parser.parse("Most Common Defect").unwrap();
        assert_eq!(q2.query_type, QueryType::MostCommonDefect);
    }

    #[test]
    fn test_unknown_with_original_string() {
        let parser = QueryParser::new();
        let q = parser.parse("random query").unwrap();

        if let QueryType::Unknown(original) = q.query_type {
            assert_eq!(original, "random query");
        } else {
            panic!("Expected Unknown query type");
        }
    }

    #[test]
    fn test_query_type_clone() {
        let qt1 = QueryType::MostCommonDefect;
        let qt2 = qt1.clone();
        assert_eq!(qt1, qt2);
    }

    #[test]
    fn test_query_struct_clone() {
        let q1 = Query {
            query_type: QueryType::ListAll,
            original: "test".to_string(),
        };
        let q2 = q1.clone();
        assert_eq!(q2.original, "test");
    }
}

// GitHub API integration module
// Toyota Way: Start simple, validate with real usage

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use octocrab::Octocrab;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Simplified repository information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoInfo {
    pub name: String,
    pub full_name: String,
    pub description: Option<String>,
    pub language: Option<String>,
    pub stars: u32,
    pub default_branch: String,
    pub updated_at: DateTime<Utc>,
}

/// GitHub organization miner
/// Phase 1: Basic organization and repository fetching
pub struct GitHubMiner {
    client: Octocrab,
}

impl GitHubMiner {
    /// Create a new GitHub miner
    ///
    /// # Arguments
    /// * `token` - Optional GitHub personal access token for authenticated requests
    ///
    /// # Examples
    /// ```no_run
    /// use organizational_intelligence_plugin::github::GitHubMiner;
    ///
    /// // Public repos only (unauthenticated)
    /// let miner = GitHubMiner::new(None);
    ///
    /// // With authentication (higher rate limits)
    /// let miner_auth = GitHubMiner::new(Some("ghp_token".to_string()));
    /// ```
    pub fn new(token: Option<String>) -> Self {
        let client = if let Some(token) = token {
            debug!("Initializing GitHub client with authentication");
            Octocrab::builder()
                .personal_token(token)
                .build()
                .expect("Failed to build Octocrab client")
        } else {
            debug!("Initializing GitHub client without authentication");
            Octocrab::builder()
                .build()
                .expect("Failed to build Octocrab client")
        };

        Self { client }
    }

    /// Fetch all repositories for an organization
    ///
    /// # Arguments
    /// * `org_name` - GitHub organization name
    ///
    /// # Errors
    /// Returns error if:
    /// - Organization name is empty
    /// - API request fails
    /// - Organization doesn't exist
    ///
    /// # Examples
    /// ```no_run
    /// use organizational_intelligence_plugin::github::GitHubMiner;
    ///
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let miner = GitHubMiner::new(None);
    /// let repos = miner.fetch_organization_repos("rust-lang").await?;
    /// println!("Found {} repositories", repos.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn fetch_organization_repos(&self, org_name: &str) -> Result<Vec<RepoInfo>> {
        // Validation: Empty org name
        if org_name.trim().is_empty() {
            return Err(anyhow!("Organization name cannot be empty"));
        }

        info!("Fetching repositories for organization: {}", org_name);

        // Fetch organization repositories
        let repos = self
            .client
            .orgs(org_name)
            .list_repos()
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch repositories for {}: {}", org_name, e))?;

        debug!("Found {} repositories for {}", repos.items.len(), org_name);

        // Convert to our simplified RepoInfo structure
        let repo_infos: Vec<RepoInfo> = repos
            .items
            .into_iter()
            .map(|repo| RepoInfo {
                name: repo.name,
                full_name: repo.full_name.unwrap_or_default(),
                description: repo.description,
                language: repo.language.and_then(|v| v.as_str().map(String::from)),
                stars: repo.stargazers_count.unwrap_or(0),
                default_branch: repo.default_branch.unwrap_or_else(|| "main".to_string()),
                updated_at: repo.updated_at.unwrap_or_else(Utc::now),
            })
            .collect();

        info!(
            "Successfully fetched {} repositories for {}",
            repo_infos.len(),
            org_name
        );

        Ok(repo_infos)
    }

    /// Filter repositories by last update date
    ///
    /// # Arguments
    /// * `repos` - List of repositories to filter
    /// * `since` - Only include repos updated since this date
    ///
    /// # Returns
    /// Filtered list of repositories
    pub fn filter_by_date(repos: Vec<RepoInfo>, since: DateTime<Utc>) -> Vec<RepoInfo> {
        repos
            .into_iter()
            .filter(|repo| repo.updated_at >= since)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_github_miner_creation() {
        // Test that GitHubMiner can be created
        let _miner = GitHubMiner::new(None);
        let _miner_with_token = GitHubMiner::new(Some("test_token".to_string()));
    }

    #[tokio::test]
    async fn test_empty_org_name_validation() {
        let miner = GitHubMiner::new(None);
        let result = miner.fetch_organization_repos("").await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[tokio::test]
    async fn test_whitespace_org_name_validation() {
        let miner = GitHubMiner::new(None);
        let result = miner.fetch_organization_repos("   ").await;

        assert!(result.is_err());
    }

    #[test]
    fn test_repo_info_structure() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "test-repo".to_string(),
            full_name: "owner/test-repo".to_string(),
            description: Some("A test repository".to_string()),
            language: Some("Rust".to_string()),
            stars: 42,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(repo.description, Some("A test repository".to_string()));
        assert_eq!(repo.language, Some("Rust".to_string()));
        assert_eq!(repo.stars, 42);
        assert_eq!(repo.default_branch, "main");
        assert_eq!(repo.updated_at, now);
    }

    #[test]
    fn test_repo_info_serialization() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "test".to_string(),
            full_name: "owner/test".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        let json = serde_json::to_string(&repo).unwrap();
        let deserialized: RepoInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(repo.name, deserialized.name);
        assert_eq!(repo.full_name, deserialized.full_name);
        assert_eq!(repo.stars, deserialized.stars);
    }

    #[test]
    fn test_filter_by_date_includes_recent() {
        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);
        let last_week = now - chrono::Duration::days(7);

        let repos = vec![
            RepoInfo {
                name: "recent".to_string(),
                full_name: "org/recent".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now,
            },
            RepoInfo {
                name: "old".to_string(),
                full_name: "org/old".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: last_week,
            },
        ];

        let filtered = GitHubMiner::filter_by_date(repos, yesterday);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "recent");
    }

    #[test]
    fn test_filter_by_date_excludes_old() {
        let now = Utc::now();
        let two_days_ago = now - chrono::Duration::days(2);
        let one_week_ago = now - chrono::Duration::days(7);

        let repos = vec![RepoInfo {
            name: "old".to_string(),
            full_name: "org/old".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: one_week_ago,
        }];

        let filtered = GitHubMiner::filter_by_date(repos, two_days_ago);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_by_date_empty_input() {
        let now = Utc::now();
        let repos: Vec<RepoInfo> = vec![];

        let filtered = GitHubMiner::filter_by_date(repos, now);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_by_date_exact_match() {
        let now = Utc::now();

        let repos = vec![RepoInfo {
            name: "exact".to_string(),
            full_name: "org/exact".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        }];

        // Filter with exact same timestamp - should be included (>=)
        let filtered = GitHubMiner::filter_by_date(repos, now);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_repo_info_with_all_fields() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "full-repo".to_string(),
            full_name: "owner/full-repo".to_string(),
            description: Some("Complete description".to_string()),
            language: Some("Python".to_string()),
            stars: 1000,
            default_branch: "develop".to_string(),
            updated_at: now,
        };

        assert!(repo.description.is_some());
        assert!(repo.language.is_some());
        assert!(repo.stars > 0);
    }

    #[test]
    fn test_repo_info_minimal_fields() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "minimal".to_string(),
            full_name: "org/minimal".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        assert!(repo.description.is_none());
        assert!(repo.language.is_none());
        assert_eq!(repo.stars, 0);
    }

    #[test]
    fn test_filter_by_date_multiple_repos() {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::days(3);

        let repos = vec![
            RepoInfo {
                name: "repo1".to_string(),
                full_name: "org/repo1".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now,
            },
            RepoInfo {
                name: "repo2".to_string(),
                full_name: "org/repo2".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now - chrono::Duration::days(2),
            },
            RepoInfo {
                name: "repo3".to_string(),
                full_name: "org/repo3".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now - chrono::Duration::days(5),
            },
        ];

        let filtered = GitHubMiner::filter_by_date(repos, cutoff);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].name, "repo1");
        assert_eq!(filtered[1].name, "repo2");
    }

    #[test]
    fn test_repo_info_clone() {
        let now = Utc::now();
        let original = RepoInfo {
            name: "test".to_string(),
            full_name: "org/test".to_string(),
            description: Some("desc".to_string()),
            language: Some("Rust".to_string()),
            stars: 100,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.full_name, cloned.full_name);
        assert_eq!(original.description, cloned.description);
        assert_eq!(original.language, cloned.language);
        assert_eq!(original.stars, cloned.stars);
        assert_eq!(original.default_branch, cloned.default_branch);
        assert_eq!(original.updated_at, cloned.updated_at);
    }

    #[test]
    fn test_repo_info_debug_format() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "test-repo".to_string(),
            full_name: "owner/test-repo".to_string(),
            description: Some("Test description".to_string()),
            language: Some("Rust".to_string()),
            stars: 42,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        let debug_str = format!("{:?}", repo);
        assert!(debug_str.contains("test-repo"));
        assert!(debug_str.contains("owner/test-repo"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_repo_info_with_empty_strings() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "".to_string(),
            full_name: "".to_string(),
            description: Some("".to_string()),
            language: Some("".to_string()),
            stars: 0,
            default_branch: "".to_string(),
            updated_at: now,
        };

        assert_eq!(repo.name, "");
        assert_eq!(repo.full_name, "");
        assert_eq!(repo.description, Some("".to_string()));
        assert_eq!(repo.language, Some("".to_string()));
    }

    #[test]
    fn test_repo_info_with_high_stars() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "popular".to_string(),
            full_name: "org/popular".to_string(),
            description: None,
            language: None,
            stars: 999999,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        assert_eq!(repo.stars, 999999);
    }

    #[test]
    fn test_repo_info_with_different_branches() {
        let now = Utc::now();

        let main_repo = RepoInfo {
            name: "main-branch".to_string(),
            full_name: "org/main-branch".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        let master_repo = RepoInfo {
            name: "master-branch".to_string(),
            full_name: "org/master-branch".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "master".to_string(),
            updated_at: now,
        };

        let develop_repo = RepoInfo {
            name: "develop-branch".to_string(),
            full_name: "org/develop-branch".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "develop".to_string(),
            updated_at: now,
        };

        assert_eq!(main_repo.default_branch, "main");
        assert_eq!(master_repo.default_branch, "master");
        assert_eq!(develop_repo.default_branch, "develop");
    }

    #[test]
    fn test_filter_by_date_preserves_order() {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::days(10);

        let repos = vec![
            RepoInfo {
                name: "first".to_string(),
                full_name: "org/first".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now,
            },
            RepoInfo {
                name: "second".to_string(),
                full_name: "org/second".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now - chrono::Duration::days(5),
            },
            RepoInfo {
                name: "third".to_string(),
                full_name: "org/third".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now - chrono::Duration::days(3),
            },
        ];

        let filtered = GitHubMiner::filter_by_date(repos, cutoff);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].name, "first");
        assert_eq!(filtered[1].name, "second");
        assert_eq!(filtered[2].name, "third");
    }

    #[test]
    fn test_repo_info_with_long_description() {
        let now = Utc::now();
        let long_desc = "A".repeat(1000);

        let repo = RepoInfo {
            name: "described".to_string(),
            full_name: "org/described".to_string(),
            description: Some(long_desc.clone()),
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        assert_eq!(repo.description, Some(long_desc));
    }

    #[test]
    fn test_filter_by_date_with_future_date() {
        let now = Utc::now();
        let future = now + chrono::Duration::days(365);

        let repos = vec![RepoInfo {
            name: "current".to_string(),
            full_name: "org/current".to_string(),
            description: None,
            language: None,
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        }];

        // Filtering with future date should exclude current repos
        let filtered = GitHubMiner::filter_by_date(repos, future);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_repo_info_deserialization() {
        let json = r#"{
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "description": "Test description",
            "language": "Rust",
            "stars": 123,
            "default_branch": "main",
            "updated_at": "2024-01-01T00:00:00Z"
        }"#;

        let repo: RepoInfo = serde_json::from_str(json).unwrap();

        assert_eq!(repo.name, "test-repo");
        assert_eq!(repo.full_name, "owner/test-repo");
        assert_eq!(repo.description, Some("Test description".to_string()));
        assert_eq!(repo.language, Some("Rust".to_string()));
        assert_eq!(repo.stars, 123);
        assert_eq!(repo.default_branch, "main");
    }

    #[test]
    fn test_repo_info_with_special_characters() {
        let now = Utc::now();
        let repo = RepoInfo {
            name: "repo-with_special.chars".to_string(),
            full_name: "org/repo-with_special.chars".to_string(),
            description: Some("Description with émojis 🚀 and special chars: <>&\"'".to_string()),
            language: Some("C++".to_string()),
            stars: 0,
            default_branch: "main".to_string(),
            updated_at: now,
        };

        assert!(repo.description.unwrap().contains("🚀"));
        assert_eq!(repo.language.unwrap(), "C++");
    }

    #[test]
    fn test_multiple_languages() {
        let now = Utc::now();

        let languages = vec![
            "Rust",
            "Python",
            "JavaScript",
            "Go",
            "TypeScript",
            "C++",
            "Java",
        ];

        for lang in languages {
            let repo = RepoInfo {
                name: format!("{}-repo", lang.to_lowercase()),
                full_name: format!("org/{}-repo", lang.to_lowercase()),
                description: None,
                language: Some(lang.to_string()),
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: now,
            };

            assert_eq!(repo.language, Some(lang.to_string()));
        }
    }

    #[tokio::test]
    async fn test_github_miner_with_empty_token() {
        let _miner = GitHubMiner::new(Some("".to_string()));
        // Should create miner even with empty token string
    }

    #[test]
    fn test_filter_by_date_all_old() {
        let now = Utc::now();
        let very_old = now - chrono::Duration::days(365);

        let repos = vec![
            RepoInfo {
                name: "old1".to_string(),
                full_name: "org/old1".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: very_old,
            },
            RepoInfo {
                name: "old2".to_string(),
                full_name: "org/old2".to_string(),
                description: None,
                language: None,
                stars: 0,
                default_branch: "main".to_string(),
                updated_at: very_old - chrono::Duration::days(30),
            },
        ];

        let filtered = GitHubMiner::filter_by_date(repos, now);
        assert_eq!(filtered.len(), 0);
    }
}

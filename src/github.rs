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
}

import { test, expect } from '@playwright/test';

// Semantic clustering tests need longer timeouts for ML computations
test.describe('Semantic Clustering', () => {
  test.setTimeout(30000); // 30s per test

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WASM to initialize
    await page.waitForTimeout(3000);
    // Switch to semantic tab
    await page.click('.tab:has-text("Semantic Clustering")');
  });

  test('should run clustering on commit messages', async ({ page }) => {
    await page.click('button:has-text("Run Clustering")');

    // Results should be visible (ML computation may take time)
    const results = page.locator('#semantic-results');
    await expect(results).toBeVisible({ timeout: 20000 });

    // Should show stats
    const stats = page.locator('#semantic-stats');
    await expect(stats).toContainText('Messages');
    await expect(stats).toContainText('Clusters');
  });

  test('should display cluster cards', async ({ page }) => {
    await page.click('button:has-text("Run Clustering")');

    // Wait for results
    await expect(page.locator('#semantic-results')).toBeVisible({ timeout: 20000 });

    // Should have cluster cards
    const clusterCards = page.locator('.cluster-card');
    expect(await clusterCards.count()).toBeGreaterThan(0);

    // Each card should have title and items
    const firstCard = clusterCards.first();
    await expect(firstCard.locator('.cluster-title')).toBeVisible();
    await expect(firstCard.locator('.cluster-items')).toBeVisible();
  });

  test('should show centroid coordinates', async ({ page }) => {
    await page.click('button:has-text("Run Clustering")');

    await expect(page.locator('#semantic-results')).toBeVisible({ timeout: 20000 });

    // Centroid chart should have coordinates
    const centroids = page.locator('#centroid-chart');
    await expect(centroids).toContainText('Cluster');
    await expect(centroids).toContainText('(');
  });

  test('should compute similarity matrix', async ({ page }) => {
    await page.click('button:has-text("Compute Similarity")');

    // Similarity section should be visible
    const similarity = page.locator('#similarity-section');
    await expect(similarity).toBeVisible({ timeout: 10000 });

    // Should have grid cells
    const cells = page.locator('.sim-cell');
    expect(await cells.count()).toBeGreaterThan(0);
  });

  test('should allow changing cluster count', async ({ page }) => {
    // Change cluster count
    const input = page.locator('#n-clusters');
    await input.fill('3');

    await page.click('button:has-text("Run Clustering")');

    // Should show 3 clusters in stats
    const stats = page.locator('#semantic-stats');
    await expect(stats).toContainText('3');
  });

  test('should clear semantic results', async ({ page }) => {
    // First run clustering
    await page.click('button:has-text("Run Clustering")');
    await expect(page.locator('#semantic-results')).toBeVisible({ timeout: 20000 });

    // Then clear - use the Clear button in the visible semantic panel
    await page.locator('#panel-semantic button.secondary:has-text("Clear")').click();
    await expect(page.locator('#semantic-results')).not.toBeVisible();
  });

  test('should group similar commits together', async ({ page }) => {
    // Use distinct commit types
    const textarea = page.locator('#semantic-commits');
    await textarea.fill(`fix: null pointer exception
fix: NPE in user service
fix: null reference error
docs: update README
docs: add installation guide
docs: improve API docs`);

    await page.locator('#n-clusters').fill('2');
    await page.click('button:has-text("Run Clustering")');

    await expect(page.locator('#semantic-results')).toBeVisible({ timeout: 20000 });

    // Should have 2 clusters
    const cards = page.locator('.cluster-card');
    expect(await cards.count()).toBe(2);
  });
});

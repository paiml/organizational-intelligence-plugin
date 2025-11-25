import { test, expect } from '@playwright/test';

test.describe('Defect Classification', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WASM to initialize
    await page.waitForTimeout(2000);
  });

  test('should analyze commit messages', async ({ page }) => {
    // Click analyze button
    await page.click('button:has-text("Analyze Commits")');

    // Results should be visible
    const results = page.locator('#results');
    await expect(results).toBeVisible();

    // Should show stats
    const stats = page.locator('#stats');
    await expect(stats).toContainText('Total Commits');
    await expect(stats).toContainText('Categories');
    await expect(stats).toContainText('Avg Confidence');
  });

  test('should display distribution table', async ({ page }) => {
    await page.click('button:has-text("Analyze Commits")');

    // Distribution table should have rows
    const table = page.locator('#distribution');
    await expect(table).toBeVisible();

    // Should have header row
    await expect(table.locator('th')).toHaveCount(4);

    // Should have data rows
    const rows = table.locator('tr');
    expect(await rows.count()).toBeGreaterThan(1);
  });

  test('should show ASCII visualization', async ({ page }) => {
    await page.click('button:has-text("Analyze Commits")');

    // ASCII chart should be visible
    const chart = page.locator('#ascii-chart');
    await expect(chart).toBeVisible();

    // Should contain bar characters
    const chartText = await chart.textContent();
    expect(chartText).toContain('█');
  });

  test('should clear results', async ({ page }) => {
    // First analyze
    await page.click('button:has-text("Analyze Commits")');
    await expect(page.locator('#results')).toBeVisible();

    // Then clear
    await page.click('button:has-text("Clear")');
    await expect(page.locator('#results')).not.toBeVisible();
  });

  test('should handle custom input', async ({ page }) => {
    // Clear default and add custom commits
    const textarea = page.locator('#commits');
    await textarea.fill('fix: memory leak in parser\nfix: null pointer in handler\nfeat: new feature');

    await page.click('button:has-text("Analyze Commits")');

    // Should show 3 commits analyzed
    const stats = page.locator('#stats');
    await expect(stats).toContainText('3');
  });

  test('should categorize security issues correctly', async ({ page }) => {
    const textarea = page.locator('#commits');
    await textarea.fill('fix: sql injection vulnerability\nfix: xss in user input\nfix: authentication bypass');

    await page.click('button:has-text("Analyze Commits")');

    // Should have security category (case insensitive)
    const table = page.locator('#distribution');
    await expect(table).toContainText(/security/i);
  });
});

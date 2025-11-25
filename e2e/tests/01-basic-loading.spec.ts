import { test, expect } from '@playwright/test';

test.describe('Basic Loading', () => {
  test('should load the OIP application', async ({ page }) => {
    await page.goto('/');

    // Wait for page to load
    await expect(page).toHaveTitle(/OIP/);

    // Check for main heading
    const heading = page.locator('h1');
    await expect(heading).toContainText('Organizational Intelligence Plugin');
  });

  test('should display WASM and SIMD badges', async ({ page }) => {
    await page.goto('/');

    // WASM badge should always be visible
    const wasmBadge = page.locator('.badge-wasm');
    await expect(wasmBadge).toBeVisible();
    await expect(wasmBadge).toHaveText('WASM');

    // SIMD badge visibility depends on browser support
    const simdBadge = page.locator('#simd-badge');
    await expect(simdBadge).toBeAttached();
  });

  test('should have working tab navigation', async ({ page }) => {
    await page.goto('/');

    // Default tab should be Defect Classification
    const classifyTab = page.locator('.tab').first();
    await expect(classifyTab).toHaveClass(/active/);

    // Click semantic tab
    const semanticTab = page.locator('.tab').nth(1);
    await semanticTab.click();
    await expect(semanticTab).toHaveClass(/active/);

    // Semantic panel should be visible
    const semanticPanel = page.locator('#panel-semantic');
    await expect(semanticPanel).toHaveClass(/active/);
  });

  test('should initialize WASM successfully', async ({ page }) => {
    await page.goto('/');

    // Wait for WASM initialization by checking console
    const consoleMessages: string[] = [];
    page.on('console', msg => consoleMessages.push(msg.text()));

    // Give WASM time to initialize
    await page.waitForTimeout(2000);

    // Check that WASM initialized
    expect(consoleMessages.some(m => m.includes('OIP WASM initialized'))).toBeTruthy();
  });
});

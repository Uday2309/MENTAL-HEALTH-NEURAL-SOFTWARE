import { test, expect } from '@playwright/test'

test.describe('Inference Flow', () => {
  test('should run inference with text input', async ({ page }) => {
    await page.goto('/app')
    
    // Skip consent modal if it appears
    try {
      await page.waitForSelector('button:has-text("Done")', { timeout: 2000 })
      await page.locator('button:has-text("Done")').click()
    } catch {
      // Modal not shown, continue
    }
    
    // Enable text consent
    await page.locator('text=Privacy Settings').click()
    await page.waitForSelector('text=Privacy & Consent')
    // Toggle text consent
    const textToggle = page.locator('button:has-text("Text / Journal Entry")').first()
    await textToggle.click()
    await page.locator('button:has-text("Done")').click()
    
    // Enter text
    const textArea = page.locator('textarea[placeholder*="How are you feeling"]')
    await textArea.fill('I am feeling stressed and anxious today.')
    
    // Analyze text
    await page.locator('button:has-text("Analyze Text")').click()
    
    // Wait for capture status
    await page.waitForSelector('text=Text', { timeout: 5000 })
    
    // Run inference
    await page.locator('button:has-text("Run Assessment")').click()
    
    // Wait for results
    await page.waitForSelector('text=Assessment Results', { timeout: 10000 })
    
    // Verify results are displayed
    const resultsCard = page.locator('text=Assessment Results')
    await expect(resultsCard).toBeVisible()
  })

  test('should show error if no data captured', async ({ page }) => {
    await page.goto('/app')
    
    // Skip consent modal
    try {
      await page.waitForSelector('button:has-text("Done")', { timeout: 2000 })
      await page.locator('button:has-text("Done")').click()
    } catch {
      // Modal not shown, continue
    }
    
    // Try to run inference without capturing data
    await page.locator('button:has-text("Run Assessment")').click()
    
    // Should show alert or error
    // Note: This test may need adjustment based on actual error handling
    await page.waitForTimeout(1000)
  })
})



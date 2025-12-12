import { test, expect } from '@playwright/test'

test.describe('Consent Flow', () => {
  test('should show consent modal on first visit', async ({ page }) => {
    await page.goto('/app')
    
    // Check if consent modal is visible
    const consentModal = page.locator('text=Privacy & Consent')
    await expect(consentModal).toBeVisible()
  })

  test('should allow toggling consent options', async ({ page }) => {
    await page.goto('/app')
    
    // Wait for consent modal
    await page.waitForSelector('text=Privacy & Consent')
    
    // Toggle vision consent
    const visionToggle = page.locator('button:has-text("Camera / Face Analysis")').first()
    await visionToggle.click()
    
    // Close modal
    const doneButton = page.locator('button:has-text("Done")')
    await doneButton.click()
    
    // Verify modal is closed
    await expect(page.locator('text=Privacy & Consent')).not.toBeVisible()
  })

  test('should persist consent state', async ({ page }) => {
    await page.goto('/app')
    
    // Set consent
    await page.waitForSelector('text=Privacy & Consent')
    const visionToggle = page.locator('button:has-text("Camera / Face Analysis")').first()
    await visionToggle.click()
    await page.locator('button:has-text("Done")').click()
    
    // Reload page
    await page.reload()
    
    // Check if consent is persisted (modal should not show if consent was given)
    // This is a simplified check - in production, you'd verify localStorage
    await expect(page.locator('text=Privacy & Consent')).not.toBeVisible({ timeout: 2000 })
  })
})



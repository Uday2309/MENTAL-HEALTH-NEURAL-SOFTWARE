import type { ConsentState } from '@mindwatch/shared'

const CONSENT_STORAGE_KEY = 'mindwatch_consent'

export function getConsentState(): ConsentState {
  if (typeof window === 'undefined') {
    return {
      vision: false,
      audio: false,
      text: false,
      context: false,
      logs: false,
    }
  }

  try {
    const stored = localStorage.getItem(CONSENT_STORAGE_KEY)
    if (stored) {
      return JSON.parse(stored) as ConsentState
    }
  } catch (error) {
    console.error('Failed to load consent state:', error)
  }

  return {
    vision: false,
    audio: false,
    text: false,
    context: false,
    logs: false,
  }
}

export function saveConsentState(consent: ConsentState): void {
  if (typeof window === 'undefined') return

  try {
    localStorage.setItem(CONSENT_STORAGE_KEY, JSON.stringify(consent))
  } catch (error) {
    console.error('Failed to save consent state:', error)
  }
}

export function clearConsentState(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(CONSENT_STORAGE_KEY)
}



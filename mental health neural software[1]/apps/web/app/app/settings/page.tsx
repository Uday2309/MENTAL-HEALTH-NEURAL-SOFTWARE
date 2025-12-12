'use client'

import { useState, useEffect } from 'react'
import { getConsentState, saveConsentState, clearConsentState } from '@/lib/consent'
import type { ConsentState } from '@mindwatch/shared'
import Link from 'next/link'

export default function SettingsPage() {
  const [consent, setConsent] = useState<ConsentState>(getConsentState())
  const [exportData, setExportData] = useState<string | null>(null)

  useEffect(() => {
    setConsent(getConsentState())
  }, [])

  const handleToggle = (key: keyof ConsentState) => {
    const newConsent = { ...consent, [key]: !consent[key] }
    setConsent(newConsent)
    saveConsentState(newConsent)
  }

  const handleExportData = () => {
    // Export consent state and any stored data
    const data = {
      consent,
      exportedAt: new Date().toISOString(),
    }
    const json = JSON.stringify(data, null, 2)
    setExportData(json)

    // Download as file
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `mindwatch-data-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleDeleteData = () => {
    if (
      confirm(
        'Are you sure you want to delete all stored data? This cannot be undone.'
      )
    ) {
      clearConsentState()
      setConsent({
        vision: false,
        audio: false,
        text: false,
        context: false,
        logs: false,
      })
      alert('All data has been deleted.')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between bg-white rounded-lg shadow-lg p-4">
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <Link
            href="/app"
            className="text-indigo-600 hover:text-indigo-700 font-medium"
          >
            Back to Dashboard
          </Link>
        </div>

        {/* Privacy Settings */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Privacy Settings
          </h2>
          <div className="space-y-4">
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Allow Research Logs (Anonymized Embeddings)
                </label>
                <button
                  onClick={() => handleToggle('logs')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    consent.logs ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      consent.logs ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <p className="text-sm text-gray-600">
                If enabled, anonymized embeddings may be stored for research
                purposes. This is completely optional and can be disabled at any
                time.
              </p>
            </div>
          </div>
        </div>

        {/* Data Management */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Data Management
          </h2>
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Export Data
              </h3>
              <p className="text-sm text-gray-600 mb-3">
                Download all your stored data in JSON format.
              </p>
              <button
                onClick={handleExportData}
                className="bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
              >
                Export Data
              </button>
              {exportData && (
                <div className="mt-4 bg-gray-50 rounded-lg p-4">
                  <pre className="text-xs overflow-auto max-h-64">
                    {exportData}
                  </pre>
                </div>
              )}
            </div>

            <div className="border-t pt-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Delete All Data
              </h3>
              <p className="text-sm text-gray-600 mb-3">
                Permanently delete all stored data. This action cannot be undone.
              </p>
              <button
                onClick={handleDeleteData}
                className="bg-red-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-red-700 transition-colors"
              >
                Delete All Data
              </button>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">About</h2>
          <div className="space-y-2 text-gray-600">
            <p>
              MindWatch is a privacy-aware, real-time student well-being
              assessment tool.
            </p>
            <p>
              All data processing happens on your device first. Only anonymized
              feature vectors are sent to our server. You control what data is
              collected.
            </p>
            <p className="text-sm text-gray-500 mt-4">
              Version 0.1.0 (MVP)
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}



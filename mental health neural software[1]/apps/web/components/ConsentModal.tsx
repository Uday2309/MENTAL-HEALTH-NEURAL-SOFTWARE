'use client'

import { useState, useEffect } from 'react'
import type { ConsentState } from '@mindwatch/shared'
import { getConsentState, saveConsentState } from '@/lib/consent'

interface ConsentModalProps {
  isOpen: boolean
  onClose: () => void
  onConsentChange: (consent: ConsentState) => void
}

export default function ConsentModal({
  isOpen,
  onClose,
  onConsentChange,
}: ConsentModalProps) {
  const [consent, setConsent] = useState<ConsentState>(getConsentState())

  useEffect(() => {
    if (isOpen) {
      setConsent(getConsentState())
    }
  }, [isOpen])

  const handleToggle = (key: keyof ConsentState) => {
    const newConsent = { ...consent, [key]: !consent[key] }
    setConsent(newConsent)
    saveConsentState(newConsent)
    onConsentChange(newConsent)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Privacy & Consent
          </h2>
          <p className="text-gray-600 mb-6">
            MindWatch respects your privacy. All data processing happens on your
            device first. Only anonymized features are sent to our server. You
            control what data is collected.
          </p>

          <div className="space-y-4">
            {/* Vision Consent */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Camera / Face Analysis
                </label>
                <button
                  onClick={() => handleToggle('vision')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    consent.vision ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      consent.vision ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <p className="text-sm text-gray-600">
                Captures facial landmarks (eye aspect ratio, mouth openness,
                head pose) on-device. No images are uploaded. Only anonymized
                feature vectors are sent.
              </p>
            </div>

            {/* Audio Consent */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Microphone / Voice Analysis
                </label>
                <button
                  onClick={() => handleToggle('audio')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    consent.audio ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      consent.audio ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <p className="text-sm text-gray-600">
                Analyzes voice features (energy, pitch, speaking rate) on-device
                using WebAudio API. No raw audio is uploaded. Only anonymized
                feature vectors are sent.
              </p>
            </div>

            {/* Text Consent */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Text / Journal Entry
                </label>
                <button
                  onClick={() => handleToggle('text')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    consent.text ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      consent.text ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <p className="text-sm text-gray-600">
                Analyzes sentiment and linguistic features on-device. Your text
                is not uploaded. Only anonymized feature vectors are sent.
              </p>
            </div>

            {/* Context Consent */}
            <div className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Location & Context
                </label>
                <button
                  onClick={() => handleToggle('context')}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    consent.context ? 'bg-indigo-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      consent.context ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <p className="text-sm text-gray-600">
                Collects coarse location (rounded to 2 decimals), time of day,
                and noise level proxy. No precise location is stored.
              </p>
            </div>

            {/* Research Logs Consent */}
            <div className="border rounded-lg p-4 bg-yellow-50">
              <div className="flex items-center justify-between mb-2">
                <label className="text-lg font-semibold text-gray-800">
                  Allow Research Logs (Optional)
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

          <div className="mt-6 flex justify-end">
            <button
              onClick={onClose}
              className="bg-indigo-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}



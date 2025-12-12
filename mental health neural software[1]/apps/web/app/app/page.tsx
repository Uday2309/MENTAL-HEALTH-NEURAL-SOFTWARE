'use client'

import { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type {
  ConsentState,
  CaptureResult,
  InferenceRequest,
  InferenceResponse,
} from '@mindwatch/shared'
import { getConsentState, saveConsentState } from '@/lib/consent'
import ConsentModal from '@/components/ConsentModal'
import VideoCapture from '@/components/VideoCapture'
import AudioCapture from '@/components/AudioCapture'
import TextBox from '@/components/TextBox'
import ContextProbe from '@/components/ContextProbe'
import ResultsCard from '@/components/ResultsCard'
import { useInference } from '@/components/InferenceClient'
import Link from 'next/link'

export default function AppPage() {
  const [consent, setConsent] = useState<ConsentState>(getConsentState())
  const [showConsentModal, setShowConsentModal] = useState(false)
  const [sessionId] = useState(() => uuidv4())
  const [captureResult, setCaptureResult] = useState<CaptureResult>({})
  const [noiseLevel, setNoiseLevel] = useState(0)
  const [inferenceResult, setInferenceResult] =
    useState<InferenceResponse | null>(null)

  const { infer, isLoading, error } = useInference()

  useEffect(() => {
    // Check if user has given any consent
    const hasAnyConsent =
      consent.vision || consent.audio || consent.text || consent.context
    if (!hasAnyConsent) {
      setShowConsentModal(true)
    }
  }, [])

  const handleConsentChange = (newConsent: ConsentState) => {
    setConsent(newConsent)
    saveConsentState(newConsent)
  }

  const handleVisionCapture = (embedding: any, features: any) => {
    setCaptureResult((prev) => ({
      ...prev,
      vision: { embedding, features },
    }))
  }

  const handleAudioCapture = (embedding: any, features: any) => {
    setCaptureResult((prev) => ({
      ...prev,
      audio: { embedding, features },
    }))
  }

  const handleTextCapture = (embedding: any, features: any) => {
    setCaptureResult((prev) => ({
      ...prev,
      text: { embedding, features },
    }))
  }

  const handleContextCapture = (embedding: any, features: any) => {
    setCaptureResult((prev) => ({
      ...prev,
      context: { embedding, features },
    }))
  }

  const handleRunInference = async () => {
    // Check if we have at least one modality
    const hasData =
      captureResult.vision ||
      captureResult.audio ||
      captureResult.text ||
      captureResult.context

    if (!hasData) {
      alert('Please capture at least one modality before running inference.')
      return
    }

    const request: InferenceRequest = {
      hv: captureResult.vision?.embedding,
      ha: captureResult.audio?.embedding,
      ht: captureResult.text?.embedding,
      hc: captureResult.context?.embedding,
      meta: {
        sessionId,
        consent,
        timestamp: new Date().toISOString(),
      },
    }

    try {
      const result = await infer(request)
      setInferenceResult(result)
    } catch (err) {
      console.error('Inference failed:', err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between bg-white rounded-lg shadow-lg p-4">
          <h1 className="text-3xl font-bold text-gray-900">MindWatch</h1>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setShowConsentModal(true)}
              className="text-indigo-600 hover:text-indigo-700 font-medium"
            >
              Privacy Settings
            </button>
            <Link
              href="/app/settings"
              className="text-gray-600 hover:text-gray-700 font-medium"
            >
              Settings
            </Link>
          </div>
        </div>

        {/* Consent Modal */}
        <ConsentModal
          isOpen={showConsentModal}
          onClose={() => setShowConsentModal(false)}
          onConsentChange={handleConsentChange}
        />

        {/* Capture Panel */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Capture Data
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Vision Capture */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Vision (Camera)
              </h3>
              <VideoCapture
                enabled={consent.vision}
                onCapture={handleVisionCapture}
                onError={(err) => console.error('Vision error:', err)}
              />
            </div>

            {/* Audio Capture */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Audio (Microphone)
              </h3>
              <AudioCapture
                enabled={consent.audio}
                onCapture={handleAudioCapture}
                onError={(err) => console.error('Audio error:', err)}
                onNoiseLevel={setNoiseLevel}
              />
            </div>

            {/* Text Input */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Text (Journal)
              </h3>
              <TextBox
                enabled={consent.text}
                onCapture={handleTextCapture}
              />
            </div>

            {/* Context Probe */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Context
              </h3>
              <ContextProbe
                enabled={consent.context}
                noiseLevel={noiseLevel}
                onCapture={handleContextCapture}
              />
            </div>
          </div>

          {/* Run Inference Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleRunInference}
              disabled={isLoading}
              className="bg-indigo-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Running Inference...' : 'Run Assessment'}
            </button>
          </div>

          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        {inferenceResult && (
          <ResultsCard result={inferenceResult} consent={consent} />
        )}

        {/* Capture Status */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">
            Capture Status
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  captureResult.vision ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
              <span className="text-gray-600">Vision</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  captureResult.audio ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
              <span className="text-gray-600">Audio</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  captureResult.text ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
              <span className="text-gray-600">Text</span>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  captureResult.context ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
              <span className="text-gray-600">Context</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}



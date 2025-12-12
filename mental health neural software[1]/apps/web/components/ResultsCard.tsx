'use client'

import type { InferenceResponse, ConsentState } from '@mindwatch/shared'

interface ResultsCardProps {
  result: InferenceResponse
  consent: ConsentState
}

export default function ResultsCard({ result, consent }: ResultsCardProps) {
  const getTrafficLightColor = () => {
    switch (result.label) {
      case 'GREEN':
        return 'bg-traffic-green'
      case 'AMBER':
        return 'bg-traffic-amber'
      case 'RED':
        return 'bg-traffic-red'
      default:
        return 'bg-gray-400'
    }
  }

  const getTrafficLightLabel = () => {
    switch (result.label) {
      case 'GREEN':
        return 'Low Stress'
      case 'AMBER':
        return 'Moderate Stress'
      case 'RED':
        return 'High Stress'
      default:
        return 'Unknown'
    }
  }

  const getSuggestions = () => {
    switch (result.label) {
      case 'GREEN':
        return [
          'Keep up the good work!',
          'Maintain healthy routines',
          'Stay connected with friends',
        ]
      case 'AMBER':
        return [
          'Take a short break',
          'Practice deep breathing',
          'Consider talking to someone',
        ]
      case 'RED':
        return [
          'Take immediate rest',
          'Reach out for support',
          'Consider professional help',
        ]
      default:
        return []
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Assessment Results</h2>

      {/* Traffic Light Display */}
      <div className="flex items-center justify-center space-x-4">
        <div
          className={`w-24 h-24 rounded-full ${getTrafficLightColor()} shadow-lg flex items-center justify-center`}
        >
          <span className="text-white text-4xl font-bold">
            {result.label === 'GREEN' ? '✓' : result.label === 'AMBER' ? '!' : '!'}
          </span>
        </div>
        <div>
          <div className="text-3xl font-bold text-gray-900">
            {getTrafficLightLabel()}
          </div>
          <div className="text-lg text-gray-600">
            Score: {(result.score * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="border-t pt-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Insights
        </h3>
        <ul className="space-y-1">
          {result.explanation.map((exp, idx) => (
            <li key={idx} className="text-gray-600 flex items-start">
              <span className="text-indigo-600 mr-2">•</span>
              <span>{exp}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Top Contributing Factors */}
      {result.topFactors && result.topFactors.length > 0 && (
        <div className="border-t pt-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">
            Top Contributing Factors
          </h3>
          <div className="space-y-2">
            {result.topFactors.map((factor, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <span className="text-gray-700 capitalize">{factor.modality}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-indigo-600 h-2 rounded-full"
                      style={{ width: `${factor.impact * 100}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600 w-12 text-right">
                    {(factor.impact * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modality Weights */}
      <div className="border-t pt-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Modality Contributions
        </h3>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {result.modalityWeights.v > 0 && consent.vision && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Vision</span>
              <span className="font-medium">
                {(result.modalityWeights.v * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {result.modalityWeights.a > 0 && consent.audio && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Audio</span>
              <span className="font-medium">
                {(result.modalityWeights.a * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {result.modalityWeights.t > 0 && consent.text && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Text</span>
              <span className="font-medium">
                {(result.modalityWeights.t * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {result.modalityWeights.c > 0 && consent.context && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Context</span>
              <span className="font-medium">
                {(result.modalityWeights.c * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Suggestions */}
      <div className="border-t pt-4 bg-blue-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Suggestions
        </h3>
        <ul className="space-y-1">
          {getSuggestions().map((suggestion, idx) => (
            <li key={idx} className="text-gray-700 flex items-start">
              <span className="text-blue-600 mr-2">→</span>
              <span>{suggestion}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Transparency Panel */}
      <details className="border-t pt-4">
        <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
          What data was used?
        </summary>
        <div className="mt-2 text-sm text-gray-600 space-y-1">
          <p>
            Only anonymized feature vectors were sent to the server. No raw
            images, audio, or text were uploaded.
          </p>
          <div className="mt-2">
            <strong>Used modalities:</strong>
            <ul className="list-disc list-inside ml-2">
              {consent.vision && <li>Vision (facial features)</li>}
              {consent.audio && <li>Audio (voice features)</li>}
              {consent.text && <li>Text (sentiment features)</li>}
              {consent.context && <li>Context (location, time, noise)</li>}
            </ul>
          </div>
        </div>
      </details>
    </div>
  )
}



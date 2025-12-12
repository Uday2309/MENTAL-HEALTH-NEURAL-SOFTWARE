'use client'

import { useState } from 'react'
import type { HT } from '@mindwatch/shared'
import { packTextEmbedding } from '@/lib/embeddings'
import Sentiment from 'sentiment'

interface TextBoxProps {
  enabled: boolean
  onCapture: (embedding: HT, features: any) => void
}

const sentiment = new Sentiment()

export default function TextBox({ enabled, onCapture }: TextBoxProps) {
  const [text, setText] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = () => {
    if (!text.trim() || !enabled) return

    setIsAnalyzing(true)

    // Analyze sentiment on-device
    const result = sentiment.analyze(text)

    // Extract features
    const words = text.trim().split(/\s+/)
    const wordCount = words.length
    const negationCount = (text.match(/\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b/gi) || []).length

    const features = {
      polarity: (result.score || 0) / 10, // Normalize to -1 to 1 range
      subjectivity: Math.abs(result.comparative || 0), // 0 to 1
      negationCount: Math.min(negationCount / wordCount, 1), // Normalize
      wordCount: Math.min(wordCount / 100, 1), // Normalize to 0-1 (assuming max 100 words)
    }

    const embedding = packTextEmbedding(features)
    onCapture(embedding, features)

    setIsAnalyzing(false)
  }

  if (!enabled) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-500">
        Text analysis disabled. Enable in consent settings.
      </div>
    )
  }

  return (
    <div className="border rounded-lg p-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Journal Entry / Text Input
      </label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="How are you feeling today? Share your thoughts..."
        className="w-full border border-gray-300 rounded-lg p-3 min-h-[120px] focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
        rows={5}
      />
      <button
        onClick={handleAnalyze}
        disabled={!text.trim() || isAnalyzing}
        className="mt-3 bg-indigo-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {isAnalyzing ? 'Analyzing...' : 'Analyze Text'}
      </button>
    </div>
  )
}



'use client'

import { useState } from 'react'
import type { InferenceRequest, InferenceResponse } from '@mindwatch/shared'

const API_URL = "https://mental-health-api-v2.onrender.com"


export async function runInference(
  request: InferenceRequest
): Promise<InferenceResponse> {
  const response = await fetch(`${API_URL}/infer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`Inference failed: ${error}`)
  }

  return response.json()
}


export function useInference() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResponse | null>(null)

  const infer = async (request: InferenceRequest) => {
    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await runInference(request)
      setResult(response)
      return response
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Unknown error occurred'
      setError(errorMessage)
      throw err
    } finally {
      setIsLoading(false)
    }
  }

  return {
    infer,
    isLoading,
    error,
    result,
  }
}



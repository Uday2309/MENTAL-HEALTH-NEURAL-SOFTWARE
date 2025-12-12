'use client'

import { useState, useEffect } from 'react'
import type { HC } from '@mindwatch/shared'
import { packContextEmbedding } from '@/lib/embeddings'

interface ContextProbeProps {
  enabled: boolean
  noiseLevel: number
  onCapture: (embedding: HC, features: any) => void
}

export default function ContextProbe({
  enabled,
  noiseLevel,
  onCapture,
}: ContextProbeProps) {
  const [location, setLocation] = useState<{
    latitude: number
    longitude: number
  } | null>(null)
  const [locationError, setLocationError] = useState<string | null>(null)

  useEffect(() => {
    if (enabled && !location) {
      captureContext()
    }
  }, [enabled])

  const captureContext = () => {
    // Get location (coarse, rounded to 2 decimals)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = Math.round(position.coords.latitude * 100) / 100
          const lon = Math.round(position.coords.longitude * 100) / 100
          setLocation({ latitude: lat, longitude: lon })
          setLocationError(null)

          // Get time context
          const now = new Date()
          const hourOfDay = now.getHours() + now.getMinutes() / 60

          const features = {
            latitude: lat,
            longitude: lon,
            hourOfDay,
            noiseLevel,
          }

          const embedding = packContextEmbedding(features)
          onCapture(embedding, features)
        },
        (error) => {
          setLocationError(error.message)
          // Fallback: use default location or skip
          const now = new Date()
          const hourOfDay = now.getHours() + now.getMinutes() / 60

          const features = {
            latitude: 0, // Default if location denied
            longitude: 0,
            hourOfDay,
            noiseLevel,
          }

          const embedding = packContextEmbedding(features)
          onCapture(embedding, features)
        },
        {
          enableHighAccuracy: false, // Coarse location
          timeout: 5000,
          maximumAge: 60000, // Cache for 1 minute
        }
      )
    } else {
      // No geolocation support
      const now = new Date()
      const hourOfDay = now.getHours() + now.getMinutes() / 60

      const features = {
        latitude: 0,
        longitude: 0,
        hourOfDay,
        noiseLevel,
      }

      const embedding = packContextEmbedding(features)
      onCapture(embedding, features)
    }
  }

  if (!enabled) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-500">
        Context collection disabled. Enable in consent settings.
      </div>
    )
  }

  return (
    <div className="border rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Context</h3>
      <div className="space-y-2 text-sm text-gray-600">
        {location ? (
          <>
            <div>
              <span className="font-medium">Location:</span>{' '}
              {location.latitude.toFixed(2)}, {location.longitude.toFixed(2)}
            </div>
            <div>
              <span className="font-medium">Time:</span>{' '}
              {new Date().toLocaleTimeString()}
            </div>
            <div>
              <span className="font-medium">Noise Level:</span> {Math.round(noiseLevel)}%
            </div>
          </>
        ) : (
          <div className="text-gray-500">
            {locationError
              ? `Location error: ${locationError}`
              : 'Capturing context...'}
          </div>
        )}
      </div>
    </div>
  )
}



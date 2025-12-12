'use client'

import { useState, useRef, useEffect } from 'react'
import type { HA } from '@mindwatch/shared'
import { packAudioEmbedding } from '@/lib/embeddings'

interface AudioCaptureProps {
  enabled: boolean
  onCapture: (embedding: HA, features: any) => void
  onError: (error: string) => void
  onNoiseLevel: (level: number) => void
}

export default function AudioCapture({
  enabled,
  onCapture,
  onError,
  onNoiseLevel,
}: AudioCaptureProps) {
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)

  useEffect(() => {
    if (enabled && !isCapturing) {
      startCapture()
    } else if (!enabled && isCapturing) {
      stopCapture()
    }

    return () => {
      stopCapture()
    }
  }, [enabled])

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const audioContext = new (window.AudioContext ||
        (window as any).webkitAudioContext)()
      audioContextRef.current = audioContext

      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      analyserRef.current = analyser
      source.connect(analyser)

      setIsCapturing(true)

      // Capture 3-second chunk
      setTimeout(() => {
        captureAudio()
      }, 3000)

      // Monitor audio level for noise proxy
      monitorAudioLevel()
    } catch (error) {
      onError('Failed to access microphone: ' + (error as Error).message)
    }
  }

  const stopCapture = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    analyserRef.current = null
    setIsCapturing(false)
  }

  const monitorAudioLevel = () => {
    if (!analyserRef.current) return

    const analyser = analyserRef.current
    const dataArray = new Uint8Array(analyser.frequencyBinCount)

    const updateLevel = () => {
      if (!analyserRef.current || !isCapturing) return

      analyser.getByteFrequencyData(dataArray)
      const average =
        dataArray.reduce((a, b) => a + b, 0) / dataArray.length
      const level = (average / 255) * 100
      setAudioLevel(level)
      onNoiseLevel(level)

      if (isCapturing) {
        requestAnimationFrame(updateLevel)
      }
    }

    updateLevel()
  }

  const captureAudio = () => {
    if (!analyserRef.current) return

    const analyser = analyserRef.current
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    const frequencyData = new Float32Array(bufferLength)

    analyser.getByteFrequencyData(dataArray)
    analyser.getFloatFrequencyData(frequencyData)

    // Compute audio features
    const features = extractAudioFeatures(dataArray, frequencyData)
    const embedding = packAudioEmbedding(features)
    onCapture(embedding, features)
  }

  const extractAudioFeatures = (
    frequencyData: Uint8Array,
    floatFrequencyData: Float32Array
  ) => {
    const bufferLength = frequencyData.length
    // Energy: RMS of frequency data
    const energy = Math.sqrt(
      Array.from(frequencyData).reduce((sum, val) => sum + val * val, 0) /
        frequencyData.length
    ) / 255

    // Pitch: find dominant frequency (simplified)
    let maxIndex = 0
    let maxValue = 0
    for (let i = 0; i < frequencyData.length; i++) {
      if (frequencyData[i] > maxValue) {
        maxValue = frequencyData[i]
        maxIndex = i
      }
    }
    const sampleRate = audioContextRef.current?.sampleRate || 44100
    const pitch = (maxIndex * sampleRate) / (2 * bufferLength)

    // Spectral centroid: weighted average of frequencies
    let weightedSum = 0
    let magnitudeSum = 0
    for (let i = 0; i < floatFrequencyData.length; i++) {
      const magnitude = Math.abs(floatFrequencyData[i])
      const frequency = (i * sampleRate) / (2 * bufferLength)
      weightedSum += frequency * magnitude
      magnitudeSum += magnitude
    }
    const spectralCentroid = magnitudeSum > 0 ? weightedSum / magnitudeSum : 0

    // Speaking rate: estimate from energy variations (simplified)
    const energyVariations = []
    const chunkSize = Math.floor(frequencyData.length / 10)
    for (let i = 0; i < frequencyData.length; i += chunkSize) {
      const chunk = frequencyData.slice(i, i + chunkSize)
      const chunkEnergy =
        chunk.reduce((sum, val) => sum + val, 0) / chunk.length
      energyVariations.push(chunkEnergy)
    }
    // Count peaks (simplified speaking rate proxy)
    let peaks = 0
    for (let i = 1; i < energyVariations.length - 1; i++) {
      if (
        energyVariations[i] > energyVariations[i - 1] &&
        energyVariations[i] > energyVariations[i + 1]
      ) {
        peaks++
      }
    }
    const speakingRate = peaks / (energyVariations.length / 3) // Normalize

    return {
      energy: Math.min(energy, 1),
      pitch: Math.min(pitch / 1000, 1), // Normalize to 0-1 (assuming max 1kHz)
      speakingRate: Math.min(speakingRate, 1),
      spectralCentroid: Math.min(spectralCentroid / 5000, 1), // Normalize
    }
  }

  if (!enabled) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-500">
        Microphone disabled. Enable in consent settings.
      </div>
    )
  }

  return (
    <div className="border rounded-lg p-4">
      <div className="mb-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-gray-700">
            Audio Level
          </span>
          <span className="text-sm text-gray-500">{Math.round(audioLevel)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-indigo-600 h-2 rounded-full transition-all"
            style={{ width: `${audioLevel}%` }}
          />
        </div>
      </div>
      {isCapturing && (
        <div className="text-sm text-indigo-600 text-center mt-2">
          Recording 3-second sample...
        </div>
      )}
    </div>
  )
}


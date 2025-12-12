'use client'

import { useState, useRef, useEffect } from 'react'
import type { HV } from '@mindwatch/shared'
import { packVisionEmbedding } from '@/lib/embeddings'

interface VideoCaptureProps {
  enabled: boolean
  onCapture: (embedding: HV, features: any) => void
  onError: (error: string) => void
}

export default function VideoCapture({
  enabled,
  onCapture,
  onError,
}: VideoCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const frameCountRef = useRef(0)

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
        video: { facingMode: 'user', width: 640, height: 480 },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsCapturing(true)

      // Capture every 5s for 10s (2 captures total)
      frameCountRef.current = 0
      captureIntervalRef.current = setInterval(() => {
        captureFrame()
        frameCountRef.current++
        if (frameCountRef.current >= 2) {
          stopCapture()
        }
      }, 5000)
    } catch (error) {
      onError('Failed to access camera: ' + (error as Error).message)
    }
  }

  const stopCapture = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCapturing(false)
  }

  const captureFrame = () => {
    if (!videoRef.current) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.drawImage(video, 0, 0)

    // Extract basic facial features (stub implementation)
    // In production, use MediaPipe or face-landmarks-web
    const features = extractFacialFeatures(canvas)
    const embedding = packVisionEmbedding(features)
    onCapture(embedding, features)
  }

  // Stub facial feature extraction
  // In production, replace with MediaPipe or face-landmarks-web
  const extractFacialFeatures = (canvas: HTMLCanvasElement) => {
    // Simulate feature extraction from face landmarks
    // These would come from MediaPipe Face Mesh or similar
    const imageData = canvas.getContext('2d')?.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    )

    // Stub: generate realistic-looking features
    // In production, use actual face detection
    const eyeAspectRatio = 0.25 + Math.random() * 0.1 // Normal range: 0.2-0.3
    const mouthOpenness = Math.random() * 0.3 // 0-0.3
    const browRaise = Math.random() * 0.2 // 0-0.2
    const headPose = {
      pitch: (Math.random() - 0.5) * 20, // -10 to 10 degrees
      yaw: (Math.random() - 0.5) * 30, // -15 to 15 degrees
      roll: (Math.random() - 0.5) * 10, // -5 to 5 degrees
    }

    return {
      eyeAspectRatio,
      mouthOpenness,
      browRaise,
      headPose,
    }
  }

  if (!enabled) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-500">
        Camera disabled. Enable in consent settings.
      </div>
    )
  }

  return (
    <div className="border rounded-lg overflow-hidden bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-auto"
      />
      {isCapturing && (
        <div className="bg-indigo-600 text-white p-2 text-center text-sm">
          Capturing... ({frameCountRef.current}/2)
        </div>
      )}
    </div>
  )
}



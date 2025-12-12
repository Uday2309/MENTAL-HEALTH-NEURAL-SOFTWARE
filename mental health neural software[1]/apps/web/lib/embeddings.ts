import type { HV, HA, HT, HC } from '@mindwatch/shared'

// Normalize embedding to [0, 1] range
export function normalizeEmbedding(embedding: number[]): number[] {
  const min = Math.min(...embedding)
  const max = Math.max(...embedding)
  const range = max - min || 1 // avoid division by zero
  return embedding.map((v) => (v - min) / range)
}

// Feature scaling: standardize to mean=0, std=1
export function standardizeEmbedding(embedding: number[]): number[] {
  const mean = embedding.reduce((a, b) => a + b, 0) / embedding.length
  const variance =
    embedding.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
    embedding.length
  const std = Math.sqrt(variance) || 1
  return embedding.map((v) => (v - mean) / std)
}

// Pad or truncate to target length
export function resizeEmbedding(
  embedding: number[],
  targetLength: number
): number[] {
  if (embedding.length === targetLength) return embedding
  if (embedding.length > targetLength) {
    return embedding.slice(0, targetLength)
  }
  // Pad with zeros
  return [...embedding, ...new Array(targetLength - embedding.length).fill(0)]
}

// Vision embedding: 32 dimensions
export function packVisionEmbedding(features: {
  eyeAspectRatio: number
  mouthOpenness: number
  browRaise: number
  headPose: { pitch: number; yaw: number; roll: number }
}): HV {
  const embedding: number[] = [
    features.eyeAspectRatio,
    features.mouthOpenness,
    features.browRaise,
    features.headPose.pitch,
    features.headPose.yaw,
    features.headPose.roll,
    // Add derived features
    Math.abs(features.headPose.pitch),
    Math.abs(features.headPose.yaw),
    features.eyeAspectRatio * features.mouthOpenness,
    features.browRaise * features.headPose.pitch,
    // Pad with normalized versions
    ...normalizeEmbedding([
      features.eyeAspectRatio,
      features.mouthOpenness,
      features.browRaise,
    ]),
    // Add more derived features to reach 32
    ...Array(32 - 12).fill(0).map((_, i) => {
      const base = [
        features.eyeAspectRatio,
        features.mouthOpenness,
        features.browRaise,
        features.headPose.pitch,
        features.headPose.yaw,
        features.headPose.roll,
      ]
      return base[i % base.length] * (0.5 + i * 0.1)
    }),
  ]
  return resizeEmbedding(embedding, 32) as HV
}

// Audio embedding: 32 dimensions
export function packAudioEmbedding(features: {
  energy: number
  pitch: number
  speakingRate: number
  spectralCentroid: number
}): HA {
  const embedding: number[] = [
    features.energy,
    features.pitch,
    features.speakingRate,
    features.spectralCentroid,
    // Derived features
    features.energy * features.pitch,
    features.speakingRate * features.spectralCentroid,
    Math.log(features.energy + 1),
    Math.log(features.pitch + 1),
    // Add normalized versions
    ...normalizeEmbedding([
      features.energy,
      features.pitch,
      features.speakingRate,
      features.spectralCentroid,
    ]),
    // Pad to 32
    ...Array(32 - 16).fill(0).map((_, i) => {
      const base = [
        features.energy,
        features.pitch,
        features.speakingRate,
        features.spectralCentroid,
      ]
      return base[i % base.length] * (0.3 + i * 0.05)
    }),
  ]
  return resizeEmbedding(embedding, 32) as HA
}

// Text embedding: 16 dimensions
export function packTextEmbedding(features: {
  polarity: number
  subjectivity: number
  negationCount: number
  wordCount: number
}): HT {
  const embedding: number[] = [
    features.polarity,
    features.subjectivity,
    features.negationCount,
    features.wordCount,
    // Derived features
    features.polarity * features.subjectivity,
    features.negationCount / Math.max(features.wordCount, 1),
    Math.abs(features.polarity),
    Math.log(features.wordCount + 1),
    // Normalized versions
    ...normalizeEmbedding([
      features.polarity,
      features.subjectivity,
      features.negationCount,
      features.wordCount,
    ]),
  ]
  return resizeEmbedding(embedding, 16) as HT
}

// Context embedding: 8 dimensions
export function packContextEmbedding(features: {
  latitude: number
  longitude: number
  hourOfDay: number
  noiseLevel: number
}): HC {
  // Normalize coordinates (assuming reasonable ranges)
  const normLat = (features.latitude + 90) / 180 // [-90, 90] -> [0, 1]
  const normLon = (features.longitude + 180) / 360 // [-180, 180] -> [0, 1]
  const normHour = features.hourOfDay / 24 // [0, 24] -> [0, 1]
  const normNoise = Math.min(features.noiseLevel / 100, 1) // [0, 100] -> [0, 1]

  const embedding: HC = [
    normLat,
    normLon,
    normHour,
    normNoise,
    // Time-based features
    Math.sin((normHour * 2 * Math.PI)), // circadian rhythm
    Math.cos((normHour * 2 * Math.PI)),
    // Location + time interaction
    normLat * normHour,
    normNoise * normHour,
  ]

  return embedding
}



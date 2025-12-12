// Embedding types
export type HV = number[]; // vision (len 32)
export type HA = number[]; // audio (len 32)
export type HT = number[]; // text (len 16)
export type HC = number[]; // context (len 8)

export interface ConsentState {
  vision: boolean;
  audio: boolean;
  text: boolean;
  context: boolean;
  logs: boolean; // allow research logs (anonymized embeddings)
}

export interface InferenceRequest {
  hv?: HV;
  ha?: HA;
  ht?: HT;
  hc?: HC;
  meta: {
    sessionId: string;
    consent: ConsentState;
    timestamp: string;
  };
}

export interface InferenceResponse {
  score: number; // 0-1, where 0=low stress, 1=high stress
  label: 'GREEN' | 'AMBER' | 'RED';
  explanation: string[];
  modalityWeights: {
    v: number; // vision contribution
    a: number; // audio contribution
    t: number; // text contribution
    c: number; // context contribution
  };
  topFactors: Array<{
    modality: 'vision' | 'audio' | 'text' | 'context';
    impact: number;
    description: string;
  }>;
}

export interface CaptureResult {
  vision?: {
    embedding: HV;
    features: {
      eyeAspectRatio: number;
      mouthOpenness: number;
      browRaise: number;
      headPose: { pitch: number; yaw: number; roll: number };
    };
  };
  audio?: {
    embedding: HA;
    features: {
      energy: number;
      pitch: number;
      speakingRate: number;
      spectralCentroid: number;
    };
  };
  text?: {
    embedding: HT;
    features: {
      polarity: number;
      subjectivity: number;
      negationCount: number;
      wordCount: number;
    };
  };
  context?: {
    embedding: HC;
    features: {
      latitude: number;
      longitude: number;
      hourOfDay: number;
      noiseLevel: number;
    };
  };
}

export interface SessionLog {
  id: string;
  sessionId: string;
  timestamp: Date;
  score: number;
  label: string;
  embeddings?: {
    hv?: HV;
    ha?: HA;
    ht?: HT;
    hc?: HC;
  };
  consent: ConsentState;
}



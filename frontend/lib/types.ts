export interface ShapValue {
  feature: string
  value: string
  shap: number
  direction: 'leave' | 'stay'
}

export interface PredictResponse {
  prediction: 'Leave' | 'Stay'
  confidence: number
  shap_values: ShapValue[]
}

export interface GlobalFeature {
  feature: string
  importance: number
}

export interface GlobalImportanceResponse {
  features: GlobalFeature[]
  summary: string
}

export interface ModelMetricsResponse {
  accuracy: number
  precision: number
  recall: number
  f1: number
  confusion_matrix: [[number, number], [number, number]]
}

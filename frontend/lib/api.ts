import { PredictResponse, GlobalImportanceResponse, ModelMetricsResponse } from './types'

const BASE = '/api'

export async function predict(data: Record<string, string | number>): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getGlobalImportance(): Promise<GlobalImportanceResponse> {
  const res = await fetch(`${BASE}/global-importance`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getModelMetrics(): Promise<ModelMetricsResponse> {
  const res = await fetch(`${BASE}/model-metrics`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

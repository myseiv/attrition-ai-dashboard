'use client'
import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { getGlobalImportance, getModelMetrics } from '@/lib/api'
import { GlobalImportanceResponse, ModelMetricsResponse } from '@/lib/types'
import ConfusionMatrix from '@/components/ConfusionMatrix'

export default function GlobalPage() {
  const [importance, setImportance] = useState<GlobalImportanceResponse | null>(null)
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null)
  const [metricsOpen, setMetricsOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([getGlobalImportance(), getModelMetrics()])
      .then(([imp, met]) => { setImportance(imp); setMetrics(met) })
      .catch(e => setError(e instanceof Error ? e.message : 'Failed to load'))
  }, [])

  if (error) return <div className="text-red-600 text-sm">{error}</div>
  if (!importance) return <div className="text-slate-400 text-sm">Loading...</div>

  const chartData = importance.features.slice(0, 15).map(f => ({
    feature: f.feature,
    importance: f.importance,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 mb-1">Global Feature Importance</h1>
        <p className="text-slate-500 text-sm">Which features matter most across all predictions?</p>
      </div>

      {/* Bar chart */}
      <div className="bg-white rounded-lg border p-6">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 20, left: 160, bottom: 0 }}
          >
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={150} />
            <Tooltip
              formatter={(val) => {
                const num = typeof val === 'number' ? val : parseFloat(String(val))
                return [isNaN(num) ? String(val) : num.toFixed(4), 'Mean |SHAP|'] as [string, string]
              }}
              contentStyle={{ fontSize: 12 }}
            />
            <Bar dataKey="importance" radius={2}>
              {chartData.map((entry, i) => (
                <Cell key={entry.feature} fill={`hsl(${220 - i * 10}, 70%, ${55 + i * 2}%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-5">
        <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">Model Insight</p>
        <p className="text-slate-700 text-sm leading-relaxed">{importance.summary}</p>
      </div>

      {/* Collapsible metrics */}
      <div className="bg-white rounded-lg border">
        <button
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setMetricsOpen(o => !o)}
        >
          <span className="font-semibold text-slate-700">Model Performance</span>
          <span className="text-slate-400 text-sm">{metricsOpen ? '▲ Hide' : '▼ Show'}</span>
        </button>
        {metricsOpen && metrics && (
          <div className="px-6 pb-6 space-y-6">
            <p className="text-xs text-slate-400 italic">
              This model was trained on a sample dataset for demonstration purposes.
              In production, further validation would be required.
            </p>
            <div className="grid grid-cols-4 gap-4">
              {(['accuracy', 'precision', 'recall', 'f1'] as const).map(key => (
                <div key={key} className="bg-slate-50 rounded-lg p-4 text-center">
                  <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{key}</p>
                  <p className="text-2xl font-bold text-slate-800 mt-1">
                    {(metrics[key] * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 mb-3">Confusion Matrix</p>
              <ConfusionMatrix matrix={metrics.confusion_matrix} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

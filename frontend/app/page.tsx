'use client'
import { useState } from 'react'
import { predict } from '@/lib/api'
import { PredictResponse } from '@/lib/types'
import PredictionResult from '@/components/PredictionResult'
import ShapChart from '@/components/ShapChart'

const DEFAULTS = {
  OverTime: 'No',
  Age: 36,
  MonthlyIncome: 6500,
  JobSatisfaction: 3,
  YearsAtCompany: 5,
  WorkLifeBalance: 3,
  JobLevel: 2,
  DistanceFromHome: 9,
  NumCompaniesWorked: 2,
  StockOptionLevel: 1,
}

export default function PredictPage() {
  const [form, setForm] = useState<Record<string, string | number>>(DEFAULTS)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await predict(form)
      setResult(res)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const set = (key: string, val: string | number) => setForm((f) => ({ ...f, [key]: val }))

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 mb-1">Predict Attrition</h1>
      <p className="text-slate-500 text-sm mb-6">Fill in employee attributes to get a prediction with explanation.</p>

      <div className="grid grid-cols-2 gap-8">
        {/* Left: Form */}
        <form onSubmit={handleSubmit} className="bg-white rounded-lg border p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Overtime</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.OverTime as string} onChange={e => set('OverTime', e.target.value)}>
                <option>Yes</option><option>No</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Age</span>
              <input type="number" min={18} max={60} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.Age as number} onChange={e => set('Age', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Monthly Income ($)</span>
              <input type="number" min={1000} max={20000} step={100} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.MonthlyIncome as number} onChange={e => set('MonthlyIncome', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Satisfaction (1–4)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobSatisfaction as number} onChange={e => set('JobSatisfaction', Number(e.target.value))}>
                <option value={1}>1 – Low</option><option value={2}>2 – Medium</option><option value={3}>3 – High</option><option value={4}>4 – Very High</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Years at Company</span>
              <input type="number" min={0} max={40} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.YearsAtCompany as number} onChange={e => set('YearsAtCompany', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Work-Life Balance (1–4)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.WorkLifeBalance as number} onChange={e => set('WorkLifeBalance', Number(e.target.value))}>
                <option value={1}>1 – Bad</option><option value={2}>2 – Good</option><option value={3}>3 – Better</option><option value={4}>4 – Best</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Level (1–5)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobLevel as number} onChange={e => set('JobLevel', Number(e.target.value))}>
                {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Distance from Home (km)</span>
              <input type="number" min={1} max={30} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.DistanceFromHome as number} onChange={e => set('DistanceFromHome', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Companies Worked At</span>
              <input type="number" min={0} max={9} className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.NumCompaniesWorked as number} onChange={e => set('NumCompaniesWorked', Number(e.target.value))} />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Stock Option Level (0–3)</span>
              <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.StockOptionLevel as number} onChange={e => set('StockOptionLevel', Number(e.target.value))}>
                {[0,1,2,3].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
          </div>
          <button type="submit" disabled={loading} className="w-full bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white rounded-md py-2.5 text-sm font-medium transition-colors">
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {/* Right: Results */}
        <div className="space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{error}</div>
          )}
          {!result && !error && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400">
              <p className="text-4xl mb-2">🔍</p>
              <p className="text-sm">Fill in the form and click Predict to see the explanation.</p>
            </div>
          )}
          {result && (
            <>
              <PredictionResult result={result} />
              <ShapChart shapValues={result.shap_values} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

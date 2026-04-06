'use client'
import { useState, useEffect, useRef } from 'react'
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

export default function WhatIfPage() {
  const [form, setForm] = useState<Record<string, string | number>>(DEFAULTS)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const res = await predict(form)
        setResult(res)
      } catch {
        // silently ignore during live updates
      } finally {
        setLoading(false)
      }
    }, 300)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [form])

  const set = (key: string, val: string | number) => setForm((f) => ({ ...f, [key]: val }))

  interface SliderProps {
    label: string
    name: string
    min: number
    max: number
    step?: number
  }

  const Slider = ({ label, name, min, max, step = 1 }: SliderProps) => (
    <label className="block">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">{label}</span>
        <span className="text-sm font-semibold text-slate-800">{form[name]}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={form[name] as number}
        onChange={e => set(name, Number(e.target.value))}
        className="w-full accent-slate-700"
      />
    </label>
  )

  return (
    <div>
      <h1 className="text-2xl font-bold text-slate-800 mb-1">What-If Explorer</h1>
      <p className="text-slate-500 text-sm mb-6">Adjust sliders to see how the prediction changes in real time.</p>

      <div className="grid grid-cols-2 gap-8">
        {/* Left: Controls */}
        <div className="bg-white rounded-lg border p-6 space-y-5">
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Overtime</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.OverTime as string} onChange={e => set('OverTime', e.target.value)}>
              <option>Yes</option><option>No</option>
            </select>
          </label>
          <Slider label="Age" name="Age" min={18} max={60} />
          <Slider label="Monthly Income ($)" name="MonthlyIncome" min={1000} max={20000} step={100} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Satisfaction</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobSatisfaction as number} onChange={e => set('JobSatisfaction', Number(e.target.value))}>
              <option value={1}>1 – Low</option><option value={2}>2 – Medium</option><option value={3}>3 – High</option><option value={4}>4 – Very High</option>
            </select>
          </label>
          <Slider label="Years at Company" name="YearsAtCompany" min={0} max={40} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Work-Life Balance</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.WorkLifeBalance as number} onChange={e => set('WorkLifeBalance', Number(e.target.value))}>
              <option value={1}>1 – Bad</option><option value={2}>2 – Good</option><option value={3}>3 – Better</option><option value={4}>4 – Best</option>
            </select>
          </label>
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Job Level</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.JobLevel as number} onChange={e => set('JobLevel', Number(e.target.value))}>
              {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <Slider label="Distance from Home (km)" name="DistanceFromHome" min={1} max={30} />
          <Slider label="Companies Worked At" name="NumCompaniesWorked" min={0} max={9} />
          <label className="block">
            <span className="text-xs font-medium text-slate-600 uppercase tracking-wide">Stock Option Level</span>
            <select className="mt-1 w-full border rounded-md px-3 py-2 text-sm" value={form.StockOptionLevel as number} onChange={e => set('StockOptionLevel', Number(e.target.value))}>
              {[0,1,2,3].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        </div>

        {/* Right: Live result */}
        <div className="space-y-4">
          {loading && !result && (
            <div className="bg-white rounded-lg border p-8 text-center text-slate-400 text-sm">Loading...</div>
          )}
          {result && (
            <>
              <div className="relative">
                {loading && <div className="absolute inset-0 bg-white/60 rounded-lg z-10" />}
                <PredictionResult result={result} />
              </div>
              <ShapChart shapValues={result.shap_values} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

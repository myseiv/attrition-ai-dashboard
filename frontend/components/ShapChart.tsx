'use client'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts'
import { ShapValue } from '@/lib/types'

interface Props {
  shapValues: ShapValue[]
}

interface TooltipProps {
  active?: boolean
  payload?: Array<{ payload: ShapValue }>
}

const CustomTooltip = ({ active, payload }: TooltipProps) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-white border border-slate-200 rounded p-2 text-sm shadow">
      <p className="font-medium">{d.feature}</p>
      <p className="text-slate-500">Value: {d.value}</p>
      <p className={d.shap > 0 ? 'text-red-600' : 'text-blue-600'}>
        SHAP: {d.shap > 0 ? '+' : ''}{d.shap.toFixed(4)}
      </p>
    </div>
  )
}

export default function ShapChart({ shapValues }: Props) {
  const data = [...shapValues].sort((a, b) => a.shap - b.shap)

  return (
    <div className="bg-white rounded-lg border p-4">
      <h3 className="text-sm font-semibold text-slate-700 mb-1">Why this prediction?</h3>
      <p className="text-xs text-slate-400 mb-3">
        <span className="text-blue-500 font-medium">Blue</span> = pushes toward Stay &nbsp;|&nbsp;
        <span className="text-red-500 font-medium">Red</span> = pushes toward Leave
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 20, left: 130, bottom: 0 }}
        >
          <XAxis type="number" domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 12 }}
            width={120}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Bar dataKey="shap" radius={2}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.shap > 0 ? '#ef4444' : '#3b82f6'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

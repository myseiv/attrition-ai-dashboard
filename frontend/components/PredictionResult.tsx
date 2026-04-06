import { PredictResponse } from '@/lib/types'

interface Props {
  result: PredictResponse
}

export default function PredictionResult({ result }: Props) {
  const isLeave = result.prediction === 'Leave'
  const pct = Math.round(result.confidence * 100)

  return (
    <div className="flex items-center gap-4 p-4 rounded-lg border bg-white">
      <div
        className={`px-5 py-2 rounded-md text-lg font-bold tracking-wide ${
          isLeave
            ? 'bg-amber-100 text-amber-800 border border-amber-300'
            : 'bg-green-100 text-green-800 border border-green-300'
        }`}
      >
        {isLeave ? '⚠ LEAVE' : '✓ STAY'}
      </div>
      <div>
        <p className="text-2xl font-semibold text-slate-800">{pct}% confidence</p>
        <p className="text-sm text-slate-500">
          {isLeave
            ? 'This employee is likely to leave'
            : 'This employee is likely to stay'}
        </p>
      </div>
    </div>
  )
}

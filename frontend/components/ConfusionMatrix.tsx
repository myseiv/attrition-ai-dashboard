interface Props {
  matrix: [[number, number], [number, number]]
}

// matrix layout: [[TN, FP], [FN, TP]]
const CELLS = [
  { label: 'True Negative', row: 0, col: 0, color: 'bg-green-100 text-green-800 border-green-300' },
  { label: 'False Positive', row: 0, col: 1, color: 'bg-red-100 text-red-800 border-red-300' },
  { label: 'False Negative', row: 1, col: 0, color: 'bg-red-100 text-red-800 border-red-300' },
  { label: 'True Positive', row: 1, col: 1, color: 'bg-green-100 text-green-800 border-green-300' },
]

export default function ConfusionMatrix({ matrix }: Props) {
  return (
    <div>
      <div className="flex gap-1 mb-1">
        <div className="w-24" />
        <div className="flex-1 text-center text-xs text-slate-500 font-medium">Predicted: Stay</div>
        <div className="flex-1 text-center text-xs text-slate-500 font-medium">Predicted: Leave</div>
      </div>
      {[0, 1].map((row) => (
        <div key={row} className="flex gap-1 mb-1">
          <div className="w-24 flex items-center justify-end pr-2 text-xs text-slate-500 font-medium">
            {row === 0 ? 'Actual: Stay' : 'Actual: Leave'}
          </div>
          {[0, 1].map((col) => {
            const cell = CELLS[row * 2 + col]
            return (
              <div
                key={col}
                className={`flex-1 border rounded-lg p-4 text-center ${cell.color}`}
              >
                <p className="text-2xl font-bold">{matrix[row][col]}</p>
                <p className="text-xs mt-1 font-medium">{cell.label}</p>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export default function FieldDisplay({ label, value }) {
  const hasValue =
    value !== undefined && value !== null && String(value).trim().length > 0

  return (
    <div className="bg-white/70 border border-brand-100 rounded-2xl p-4">
      <div className="text-brand-700 text-xs font-semibold">{label}</div>
      <div className="mt-2 text-brand-900 text-sm font-semibold leading-6">
        {hasValue ? value : '—'}
      </div>
    </div>
  )
}


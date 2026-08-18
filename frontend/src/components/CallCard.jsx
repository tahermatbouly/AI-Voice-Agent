import { Link } from 'react-router-dom'
import { useLocale } from '../lib/i18n'

export default function CallCard({ call }) {
  const { t, locale } = useLocale()
  const caller = call?.callerName || t.dashboard.unknownCaller
  const summary = call?.inquirySummary || call?.inquiry || ''
  const ts = call?.timestamp ? new Date(call.timestamp) : null

  const formatted = ts
    ? new Intl.DateTimeFormat(locale === 'ar' ? 'ar-EG' : 'en-US', {
        dateStyle: 'medium',
        timeStyle: 'short',
      }).format(ts)
    : '—'

  return (
    <Link
      to={`/dashboard/${call.id}`}
      className="block rounded-2xl border border-brand-100 bg-white/70 hover:bg-white/90 transition p-4 shadow-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-brand-900 font-semibold truncate">{caller}</div>
          <div className="text-brand-700 text-sm mt-1 leading-6 line-clamp-2">
            {summary || '—'}
          </div>
        </div>
        <div className="text-brand-600 text-xs font-semibold whitespace-nowrap">
          {formatted}
        </div>
      </div>
    </Link>
  )
}


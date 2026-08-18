import { useEffect, useMemo, useState } from 'react'
import CallCard from '../components/CallCard'
import { fetchCalls } from '../lib/api'
import { useLocale } from '../lib/i18n'

function formatForSearch(call) {
  const caller = call?.callerName ? String(call.callerName) : ''
  const summary = call?.inquirySummary ? String(call.inquirySummary) : ''
  const ts = call?.timestamp ? new Date(call.timestamp).toISOString() : ''
  return `${caller} ${summary} ${ts}`.toLowerCase()
}

export default function DashboardPage() {
  const { t } = useLocale()
  const [calls, setCalls] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')

  useEffect(() => {
    let mounted = true
    async function run() {
      setLoading(true)
      setError('')
      try {
        const data = await fetchCalls()
        if (mounted) setCalls(Array.isArray(data) ? data : [])
      } catch (e) {
        if (mounted) setError(t.dashboard.loadError)
      } finally {
        if (mounted) setLoading(false)
      }
    }
    run()
    return () => {
      mounted = false
    }
  }, [t.dashboard.loadError])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return calls
    return calls.filter((c) => formatForSearch(c).includes(q))
  }, [calls, query])

  return (
    <div className="w-full">
      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        <div className="flex items-start justify-between gap-4 flex-wrap mb-6">
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-brand-900">
              {t.dashboard.title}
            </h1>
            <p className="text-brand-600 text-sm mt-2 max-w-2xl">
              {t.dashboard.subtitle}
            </p>
          </div>

          <div className="w-full sm:w-80">
            <label className="text-xs font-semibold text-brand-700 block mb-2">
              {t.dashboard.search}
            </label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t.dashboard.searchPlaceholder}
              className="w-full rounded-2xl border border-brand-100 bg-white/70 px-4 py-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-500/40"
            />
          </div>
        </div>

        {loading ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="rounded-2xl border border-brand-100 bg-white/70 p-4 shadow-sm animate-pulse"
              >
                <div className="h-4 w-2/3 rounded bg-brand-200 mb-3" />
                <div className="h-3 w-full rounded bg-brand-100 mb-2" />
                <div className="h-3 w-5/6 rounded bg-brand-100" />
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="rounded-2xl border border-red-100 bg-red-50/60 px-4 py-3 text-red-700 font-semibold">
            {error}
          </div>
        ) : filtered.length === 0 ? (
          <div className="rounded-3xl border border-brand-100 bg-white/70 p-10 text-center shadow-sm">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-brand-50 border border-brand-100 flex items-center justify-center mb-3">
              <span className="text-brand-700 font-bold">—</span>
            </div>
            <div className="text-brand-900 font-extrabold text-lg">
              {t.dashboard.emptyTitle}
            </div>
            <div className="text-brand-600 text-sm mt-2">
              {t.dashboard.emptySubtitle}
            </div>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {filtered
              .slice()
              .sort((a, b) => new Date(b.timestamp).valueOf() - new Date(a.timestamp).valueOf())
              .map((call) => (
                <CallCard key={call.id} call={call} />
              ))}
          </div>
        )}
      </div>
    </div>
  )
}


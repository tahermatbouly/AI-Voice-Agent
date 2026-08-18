import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import FieldDisplay from '../components/FieldDisplay'
import TranscriptBubble from '../components/TranscriptBubble'
import { fetchCallById } from '../lib/api'
import { useLocale } from '../lib/i18n'

function formatTimestamp(ts, locale) {
  if (!ts) return '—'
  const d = new Date(ts)
  return new Intl.DateTimeFormat(locale === 'ar' ? 'ar-EG' : 'en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d)
}

export default function CallDetailPage() {
  const { t, locale } = useLocale()
  const { callId } = useParams()
  const [call, setCall] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let mounted = true
    async function run() {
      setLoading(true)
      setError('')
      try {
        const data = await fetchCallById(callId)
        if (!mounted) return
        setCall(data)
      } catch (e) {
        if (mounted) setError(t.detail.loadError)
      } finally {
        if (mounted) setLoading(false)
      }
    }
    run()
    return () => {
      mounted = false
    }
  }, [callId, t.detail.loadError])

  const transcript = useMemo(() => {
    const turns = call?.transcript
    if (Array.isArray(turns)) return turns

    // Defensive: handle alternative shapes from different backends.
    if (Array.isArray(call?.turns)) return call.turns
    if (Array.isArray(call?.messages)) return call.messages
    return []
  }, [call])

  const extracted = call || {}

  return (
    <div className="w-full">
      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        <div className="flex items-center justify-between gap-4 flex-wrap mb-6">
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 rounded-2xl border border-brand-100 bg-white/70 px-4 py-2 text-sm font-semibold shadow-sm hover:bg-white/90 transition"
          >
            <span aria-hidden="true">←</span>
            {t.detail.back}
          </Link>

          <div className="text-sm text-brand-600 font-semibold">
            {call?.timestamp ? formatTimestamp(call.timestamp, locale) : ''}
          </div>
        </div>

        {loading ? (
          <div className="grid gap-4 lg:grid-cols-3">
            <div className="lg:col-span-1 rounded-3xl border border-brand-100 bg-white/70 p-4 shadow-sm animate-pulse">
              <div className="h-6 w-2/3 rounded bg-brand-200 mb-3" />
              <div className="h-10 w-full rounded bg-brand-100" />
            </div>
            <div className="lg:col-span-2 rounded-3xl border border-brand-100 bg-white/70 p-4 shadow-sm animate-pulse">
              <div className="h-6 w-1/2 rounded bg-brand-200 mb-3" />
              <div className="h-36 w-full rounded bg-brand-100" />
            </div>
          </div>
        ) : error ? (
          <div className="rounded-2xl border border-red-100 bg-red-50/60 px-4 py-3 text-red-700 font-semibold">
            {error}
          </div>
        ) : !call ? (
          <div className="rounded-3xl border border-brand-100 bg-white/70 p-10 text-center shadow-sm">
            <div className="text-brand-900 font-extrabold text-lg">
              {t.detail.notFoundTitle}
            </div>
            <div className="text-brand-600 text-sm mt-2">
              {t.detail.notFoundSubtitle}
            </div>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-3">
            <section className="lg:col-span-1 space-y-4">
              <div className="rounded-3xl border border-brand-100 bg-white/70 p-4 shadow-sm">
                <div className="text-brand-900 font-extrabold text-lg mb-2">
                  {t.detail.extractedTitle}
                </div>
                <div className="text-brand-600 text-xs font-semibold">
                  {t.detail.extractedHint}
                </div>
              </div>

              <FieldDisplay label={t.detail.name} value={extracted.name || extracted.callerName} />
              <FieldDisplay label={t.detail.address} value={extracted.address} />
              <FieldDisplay label={t.detail.position} value={extracted.position} />
              <FieldDisplay label={t.detail.inquiry} value={extracted.inquiry} />
              <FieldDisplay label={t.detail.notes} value={extracted.notes} />
            </section>

            <section className="lg:col-span-2">
              <div className="rounded-3xl border border-brand-100 bg-white/70 p-4 shadow-sm">
                <div className="flex items-center justify-between gap-3 mb-3">
                  <div className="text-brand-900 font-extrabold text-lg">
                    {t.detail.transcript}
                  </div>
                  <div className="text-brand-600 text-xs font-semibold">
                    {transcript.length} {t.detail.messages}
                  </div>
                </div>

                <div className="max-h-[60vh] overflow-auto px-1 pb-2">
                  <div className="space-y-3">
                    {transcript.length ? (
                      transcript.map((turn, idx) => (
                        <TranscriptBubble
                          key={turn.id || idx}
                          role={turn.role || turn.speaker || 'caller'}
                          text={turn.text || turn.message || ''}
                          agentLabel={t.detail.agent}
                          callerLabel={t.detail.caller}
                        />
                      ))
                    ) : (
                      <div className="text-center text-brand-600 text-sm py-10">
                        {t.detail.noTranscript}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}


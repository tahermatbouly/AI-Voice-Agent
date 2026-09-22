import { useEffect, useMemo, useRef, useState } from 'react'
import CallButton from '../components/CallButton'
import CallStatusOrb, { CALL_STATUSES } from '../components/CallStatusOrb'
import AudioWaveform from '../components/AudioWaveform'
import FieldDisplay from '../components/FieldDisplay'
import { VoiceAgentSession } from '../lib/voiceAgent'
import { useLocale } from '../lib/i18n'

const CANDIDATE_FIELDS = [
  ['candidate_name', 'name'],
  ['target_domain', 'domain'],
  ['experience_summary', 'experience'],
  ['years_of_experience', 'years'],
  ['education_level', 'education'],
  ['key_skills', 'skills'],
  ['tools_technologies', 'tools'],
  ['english_proficiency', 'english'],
  ['notes', 'notes'],
]

function formatFieldValue(value) {
  if (value === undefined || value === null || value === '') return ''
  if (Array.isArray(value)) return value.join('، ')
  return String(value)
}

export default function CallPage() {
  const { t, locale } = useLocale()
  const [status, setStatus] = useState(CALL_STATUSES.idle)
  const [level, setLevel] = useState(0)
  const [botSpeaking, setBotSpeaking] = useState(false)
  const [error, setError] = useState('')
  const [transcript, setTranscript] = useState('')
  const [turnInfo, setTurnInfo] = useState(null)
  const [candidate, setCandidate] = useState(null)
  const [endedEarly, setEndedEarly] = useState(false)

  const sessionRef = useRef(null)
  const levelFadeRef = useRef(null)

  const isInCall = status === CALL_STATUSES.inCall
  const canStart = useMemo(
    () => status === CALL_STATUSES.idle || status === CALL_STATUSES.ended,
    [status],
  )

  useEffect(() => {
    return () => {
      if (levelFadeRef.current) clearInterval(levelFadeRef.current)
      sessionRef.current?.stop()
      sessionRef.current = null
    }
  }, [])

  function clearSessionUiTimers() {
    if (levelFadeRef.current) clearInterval(levelFadeRef.current)
    levelFadeRef.current = null
    setLevel(0)
  }

  async function handleStartCall() {
    setError('')
    setTranscript('')
    setCandidate(null)
    setEndedEarly(false)
    setTurnInfo(null)
    setBotSpeaking(false)
    setStatus(CALL_STATUSES.connecting)

    const session = new VoiceAgentSession({
      onLevel: (value) => {
        setLevel((prev) => prev * 0.55 + value * 0.45)
      },
      onBotSpeaking: setBotSpeaking,
      onTranscript: (text) => setTranscript(text),
      onTurn: (info) => setTurnInfo(info),
      onComplete: ({ candidate: data, endedEarly: early }) => {
        setCandidate(data)
        setEndedEarly(early)
        setStatus(CALL_STATUSES.ended)
        setBotSpeaking(false)
        clearSessionUiTimers()
        sessionRef.current = null
      },
      onError: (message) => {
        setError(message)
      },
      onDisconnected: () => {
        if (sessionRef.current === session) {
          setStatus((prev) =>
            prev === CALL_STATUSES.inCall ? CALL_STATUSES.ended : prev,
          )
          setBotSpeaking(false)
          clearSessionUiTimers()
          sessionRef.current = null
        }
      },
    })

    sessionRef.current = session

    try {
      await session.start()
      setStatus(CALL_STATUSES.inCall)
      setLevel(0.15)
    } catch (e) {
      session.stop()
      sessionRef.current = null
      setStatus(CALL_STATUSES.ended)
      setError(
        e instanceof Error ? e.message : t.call.connectFallbackError,
      )
    }
  }

  function handleEndCall() {
    setError('')
    sessionRef.current?.stop()
    sessionRef.current = null
    setBotSpeaking(false)
    clearSessionUiTimers()
    setStatus(CALL_STATUSES.ended)
  }

  const statusHint = botSpeaking
    ? t.call.botSpeakingHint
    : isInCall
      ? t.call.activeHint
      : t.call.idleHint

  return (
    <div className="w-full">
      <div className="mx-auto w-full max-w-6xl px-4 py-10">
        <div className="flex items-center justify-between gap-4 mb-10">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-brand-50 border border-brand-100 flex items-center justify-center shadow-sm">
              <div className="h-2.5 w-2.5 rounded-full bg-brand-500 shadow-[0_0_25px_rgba(31,139,255,0.55)]" />
            </div>
            <div className="text-start">
              <div className="text-brand-900 font-extrabold text-lg">
                {t.call.brandTitle}
              </div>
              <div className="text-brand-600 text-sm">{t.call.brandSubtitle}</div>
            </div>
          </div>
          <div className="hidden md:block text-xs text-brand-600 font-semibold">
            {statusHint}
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(280px,360px)] items-start">
          <div className="flex flex-col items-center text-center">
            <div className="animate-softFloat">
              <CallStatusOrb
                status={status}
                labels={{
                  idle: t.call.statusIdle,
                  connecting: t.call.statusConnecting,
                  inCall: botSpeaking
                    ? t.call.statusBotSpeaking
                    : t.call.statusInCall,
                  ended: t.call.statusEnded,
                }}
              />
            </div>

            <AudioWaveform
              level={level}
              active={isInCall}
              activeLabel={
                botSpeaking ? t.call.waveformBot : t.call.waveformActive
              }
              idleLabel={t.call.waveformIdle}
            />

            <div className="mt-8 w-full max-w-md flex flex-col items-center gap-3">
              {canStart ? (
                <CallButton
                  variant="start"
                  onClick={handleStartCall}
                  disabled={!canStart}
                  startLabel={t.call.start}
                  endLabel={t.call.end}
                />
              ) : null}

              {isInCall ? (
                <CallButton
                  variant="end"
                  onClick={handleEndCall}
                  disabled={!isInCall}
                  startLabel={t.call.start}
                  endLabel={t.call.end}
                />
              ) : null}

              {status === CALL_STATUSES.connecting ? (
                <div className="text-brand-600 text-sm font-semibold">
                  {t.call.connectingInline}
                </div>
              ) : null}

              {error ? (
                <div className="w-full rounded-2xl border border-red-100 bg-red-50/60 text-red-700 px-4 py-3 text-sm font-semibold text-start">
                  {error}
                </div>
              ) : null}
            </div>
          </div>

          <aside className="w-full space-y-4">
            <section className="rounded-3xl border border-brand-100 bg-white/70 backdrop-blur-sm p-5 shadow-sm text-start">
              <div className="text-brand-800 text-sm font-bold">
                {t.call.liveTitle}
              </div>
              <div className="mt-1 text-brand-500 text-xs font-semibold">
                {turnInfo?.mode
                  ? `${t.call.modeLabel}: ${turnInfo.mode}`
                  : t.call.liveHint}
              </div>

              <div
                className={[
                  'mt-4 min-h-[4.5rem] rounded-2xl border border-brand-50 bg-brand-50/50 px-4 py-3 text-sm leading-7',
                  locale === 'ar' ? 'text-right' : 'text-left',
                ].join(' ')}
                dir="auto"
              >
                {transcript ? (
                  <span className="text-brand-900 font-semibold">
                    {transcript}
                  </span>
                ) : (
                  <span className="text-brand-400">{t.call.waitingSpeech}</span>
                )}
              </div>
            </section>

            {candidate ? (
              <section className="rounded-3xl border border-brand-100 bg-white/70 backdrop-blur-sm p-5 shadow-sm text-start">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-brand-800 text-sm font-bold">
                    {t.call.resultTitle}
                  </div>
                  <div className="text-[11px] font-semibold text-brand-500">
                    {endedEarly
                      ? t.call.resultEndedEarly
                      : t.call.resultComplete}
                  </div>
                </div>

                <div className="mt-4 grid gap-3">
                  {CANDIDATE_FIELDS.map(([key, labelKey]) => {
                    const value = formatFieldValue(candidate[key])
                    if (!value) return null
                    return (
                      <FieldDisplay
                        key={key}
                        label={t.call.fields[labelKey]}
                        value={value}
                      />
                    )
                  })}
                </div>
              </section>
            ) : null}
          </aside>
        </div>
      </div>
    </div>
  )
}

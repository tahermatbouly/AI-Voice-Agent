import { useEffect, useMemo, useRef, useState } from 'react'
import CallButton from '../components/CallButton'
import CallStatusOrb, { CALL_STATUSES } from '../components/CallStatusOrb'
import AudioWaveform from '../components/AudioWaveform'
import { connectToCall } from '../lib/livekitConnect'
import { useLocale } from '../lib/i18n'

export default function CallPage() {
  const { t } = useLocale()
  const [status, setStatus] = useState(CALL_STATUSES.idle)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState('')
  const roomRef = useRef(null)
  const levelTimerRef = useRef(null)

  const isInCall = status === CALL_STATUSES.inCall

  const canStart = useMemo(() => {
    return status === CALL_STATUSES.idle || status === CALL_STATUSES.ended
  }, [status])

  useEffect(() => {
    // Cleanup if the user navigates away during an active call.
    return () => {
      if (levelTimerRef.current) clearInterval(levelTimerRef.current)
      levelTimerRef.current = null

      if (roomRef.current && typeof roomRef.current.disconnect === 'function') {
        roomRef.current.disconnect()
      }
      roomRef.current = null
    }
  }, [])

  function startMockVoiceLevel() {
    // If/when you wire LiveKit audio processing, replace this with actual audio levels.
    levelTimerRef.current = setInterval(() => {
      // Add some randomness so the visualization feels alive but not jittery.
      const base = 0.35 + Math.random() * 0.65
      const smooth = 0.6 * base + 0.4 * (Math.random() > 0.5 ? 0.15 : 0.8)
      setLevel((prev) => prev * 0.55 + smooth * 0.45)
    }, 110)
  }

  async function handleStartCall() {
    setError('')
    setStatus(CALL_STATUSES.connecting)

    try {
      // 1) Connect to LiveKit room (stubbed by default).
      const room = await connectToCall()
      roomRef.current = room

      // 2) Switch UI to "in-call" state.
      setStatus(CALL_STATUSES.inCall)
      setLevel(0.2)

      // 3) Start voice activity visualization.
      startMockVoiceLevel()
    } catch (e) {
      setStatus(CALL_STATUSES.ended)
      setError(
        e instanceof Error
          ? e.message
          : t.call.connectFallbackError
      )
    }
  }

  function handleEndCall() {
    setError('')

    if (levelTimerRef.current) clearInterval(levelTimerRef.current)
    levelTimerRef.current = null
    setLevel(0)

    if (roomRef.current && typeof roomRef.current.disconnect === 'function') {
      roomRef.current.disconnect()
    }
    roomRef.current = null

    setStatus(CALL_STATUSES.ended)
  }

  return (
    <div className="w-full">
      <div className="mx-auto w-full max-w-6xl px-4 py-10">
        {/* Branding */}
        <div className="flex items-center justify-between gap-4 mb-10">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-brand-50 border border-brand-100 flex items-center justify-center shadow-sm">
              <div className="h-2.5 w-2.5 rounded-full bg-brand-500 shadow-[0_0_25px_rgba(31,139,255,0.55)]" />
            </div>
            <div>
              <div className="text-brand-900 font-extrabold">{t.call.brandTitle}</div>
              <div className="text-brand-600 text-sm">{t.call.brandSubtitle}</div>
            </div>
          </div>
          <div className="hidden md:block text-xs text-brand-600 font-semibold">
            {isInCall ? t.call.activeHint : t.call.idleHint}
          </div>
        </div>

        <div className="flex flex-col items-center text-center">
          <div className="animate-softFloat">
            <CallStatusOrb
              status={status}
              labels={{
                idle: t.call.statusIdle,
                connecting: t.call.statusConnecting,
                inCall: t.call.statusInCall,
                ended: t.call.statusEnded,
              }}
            />
          </div>

          <AudioWaveform
            level={level}
            active={isInCall}
            activeLabel={t.call.waveformActive}
            idleLabel={t.call.waveformIdle}
          />

          <div className="mt-8 w-full max-w-md">
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
              <div className="mt-4 text-brand-600 text-sm font-semibold">
                {t.call.connectingInline}
              </div>
            ) : null}

            {error ? (
              <div className="mt-4 rounded-2xl border border-red-100 bg-red-50/60 text-red-700 px-4 py-3 text-sm font-semibold">
                {error}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}


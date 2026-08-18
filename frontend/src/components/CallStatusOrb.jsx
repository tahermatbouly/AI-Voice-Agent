const STATUS = {
  idle: 'idle',
  connecting: 'connecting',
  inCall: 'in_call',
  ended: 'ended',
}

function statusToUi(status, labels) {
  switch (status) {
    case STATUS.connecting:
      return {
        ring:
          'border-brand-400/60 shadow-[0_0_0_10px_rgba(31,139,255,0.05),0_0_45px_rgba(31,139,255,0.18)]',
        orb: 'bg-gradient-to-b from-brand-400/20 to-brand-600/25',
        label: labels.connecting,
      }
    case STATUS.inCall:
      return {
        ring:
          'border-brand-500/80 shadow-[0_0_0_10px_rgba(31,139,255,0.08),0_0_60px_rgba(31,139,255,0.55)]',
        orb: 'bg-gradient-to-b from-brand-400/35 to-brand-600/45',
        label: labels.inCall,
      }
    case STATUS.ended:
      return {
        ring: 'border-brand-200 shadow-none',
        orb: 'bg-gradient-to-b from-brand-50 to-brand-100',
        label: labels.ended,
      }
    case STATUS.idle:
    default:
      return {
        ring: 'border-brand-200 shadow-none',
        orb: 'bg-gradient-to-b from-brand-50 to-brand-100',
        label: labels.idle,
      }
  }
}

export default function CallStatusOrb({ status, labels }) {
  const ui = statusToUi(status, labels)

  return (
    <div className="relative w-full flex flex-col items-center">
      <div
        className={[
          'relative rounded-full border-2',
          ui.ring,
          status === STATUS.inCall ? 'animate-orbPulse' : '',
          status === STATUS.idle ? 'animate-softFloat' : '',
        ].join(' ')}
        style={{ width: 220, height: 220 }}
        aria-label={ui.label}
        role="status"
      >
        <div
          className={[
            'absolute inset-3 rounded-full blur-0',
            ui.orb,
            status === STATUS.inCall ? 'ring-1 ring-brand-500/40' : '',
          ].join(' ')}
        />

        {/* Connecting spinner */}
        {status === STATUS.connecting ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-10 w-10 rounded-full border-4 border-brand-200 border-t-brand-500 animate-spin" />
          </div>
        ) : null}

        {/* Voice "presence" dot */}
        {status === STATUS.inCall ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-3 w-3 rounded-full bg-white/90 shadow-[0_0_30px_rgba(255,255,255,0.65)]" />
          </div>
        ) : null}
      </div>

      <div className="mt-4 text-brand-800 text-sm font-semibold">{ui.label}</div>
    </div>
  )
}

export { STATUS as CALL_STATUSES }


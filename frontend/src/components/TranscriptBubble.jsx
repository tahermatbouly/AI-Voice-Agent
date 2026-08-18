function bubbleBase(role) {
  if (role === 'agent') {
    return {
      bubble:
        'bg-brand-600 text-white border-brand-600/30 rounded-2xl rounded-tr-md',
      pill: 'bg-white/15 border-white/15',
    }
  }

  return {
    bubble: 'bg-white/80 text-brand-900 border-brand-100 rounded-2xl rounded-tl-md',
    pill: 'bg-brand-50 border-brand-100',
  }
}

export default function TranscriptBubble({ role, text, agentLabel, callerLabel }) {
  const ui = bubbleBase(role)
  const isAgent = role === 'agent'

  return (
    <div className={`w-full flex ${isAgent ? 'justify-end' : 'justify-start'}`}>
      <div className="max-w-[80%]">
        <div
          className={[
            ui.bubble,
            'border px-4 py-3 shadow-sm',
            'transition-transform',
          ].join(' ')}
        >
          <div className="text-xs font-bold opacity-90 mb-1">
            {isAgent ? agentLabel : callerLabel}
          </div>
          <div className="text-sm leading-6 whitespace-pre-wrap">{text}</div>
        </div>
      </div>
    </div>
  )
}


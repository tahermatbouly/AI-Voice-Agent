export default function CallButton({ variant, onClick, disabled, startLabel, endLabel }) {
  const isEnd = variant === 'end'

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={[
        'w-full sm:w-auto rounded-2xl px-6 py-3 font-semibold transition',
        'border focus-visible:ring-2 focus-visible:ring-brand-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-white',
        isEnd
          ? 'bg-gradient-to-b from-red-500 to-red-600 text-white border-red-200/50 hover:from-red-600 hover:to-red-700 disabled:opacity-50'
          : 'bg-gradient-to-b from-brand-600 to-brand-700 text-white border-brand-200/50 hover:from-brand-700 hover:to-brand-800 disabled:opacity-50',
      ].join(' ')}
    >
      {isEnd ? endLabel : startLabel}
    </button>
  )
}


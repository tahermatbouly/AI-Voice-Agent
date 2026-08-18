export default function AudioWaveform({
  level = 0,
  active = false,
  activeLabel,
  idleLabel,
}) {
  // level: 0..1
  const bars = Array.from({ length: 20 }, (_, i) => i)

  return (
    <div className="w-full max-w-xl mx-auto mt-6">
      <div className="flex items-end justify-center gap-1 h-12">
        {bars.map((i) => {
          const phase = i / bars.length
          const wave =
            0.15 +
            0.85 *
              (0.55 * level +
                0.45 * (0.6 + 0.4 * Math.sin((phase + level) * Math.PI * 2)))

          const height = Math.max(0.12, Math.min(1, wave))

          return (
            <div
              key={i}
              className={[
                'w-[5px] rounded-full',
                'bg-brand-600',
                active ? 'animate-waveform' : '',
              ].join(' ')}
              style={{
                height: `${height * 100}%`,
                opacity: active ? 1 : 0.35,
              }}
            />
          )
        })}
      </div>
      <div className="mt-2 text-center text-brand-600/80 text-xs">
        {active ? activeLabel : idleLabel}
      </div>
    </div>
  )
}


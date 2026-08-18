import { NavLink } from 'react-router-dom'
import { useLocale } from '../lib/i18n'

export default function TopNav() {
  const { t, toggleLocale, isArabic } = useLocale()

  return (
    <header className="sticky top-0 z-20 bg-white/70 backdrop-blur border-b border-brand-100">
      <div className="mx-auto w-full max-w-6xl px-4">
        <div className="flex items-center justify-between py-3 gap-3">
          <div className="flex items-center gap-3">
            <div
              className="h-10 w-10 rounded-2xl bg-brand-50 border border-brand-200 flex items-center justify-center shadow-sm"
              aria-hidden="true"
            >
              <div className="h-2.5 w-2.5 rounded-full bg-brand-500 shadow-[0_0_30px_rgba(31,139,255,0.55)]" />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-brand-900 text-sm">{t.nav.brandTitle}</div>
              <div className="text-brand-600 text-xs">{t.nav.brandSubtitle}</div>
            </div>
          </div>

          <nav className="flex items-center gap-2 flex-wrap">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                [
                  'px-4 py-2 rounded-xl text-sm font-semibold transition',
                  isActive
                    ? 'bg-brand-600 text-white shadow-sm'
                    : 'bg-white/70 hover:bg-brand-50 text-brand-800 border border-brand-100',
                ].join(' ')
              }
            >
              {t.nav.call}
            </NavLink>
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                [
                  'px-4 py-2 rounded-xl text-sm font-semibold transition',
                  isActive
                    ? 'bg-brand-600 text-white shadow-sm'
                    : 'bg-white/70 hover:bg-brand-50 text-brand-800 border border-brand-100',
                ].join(' ')
              }
            >
              {t.nav.dashboard}
            </NavLink>
            <button
              type="button"
              onClick={toggleLocale}
              className="px-3 py-2 rounded-xl text-sm font-semibold transition bg-white/70 hover:bg-brand-50 text-brand-800 border border-brand-100 min-w-14"
              aria-label={isArabic ? 'Switch to English' : 'التبديل إلى العربية'}
            >
              {t.nav.switchLabel}
            </button>
          </nav>
        </div>
      </div>
    </header>
  )
}


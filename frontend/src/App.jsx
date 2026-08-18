import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import TopNav from './components/TopNav'
import CallPage from './pages/CallPage'
import DashboardPage from './pages/DashboardPage'
import CallDetailPage from './pages/CallDetailPage'
import { LocaleProvider } from './lib/i18n'

export default function App() {
  return (
    <LocaleProvider>
      <BrowserRouter>
        <div className="min-h-screen w-full bg-transparent">
          <TopNav />
          <main className="w-full">
            <Routes>
              <Route path="/" element={<CallPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/dashboard/:callId" element={<CallDetailPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </LocaleProvider>
  )
}

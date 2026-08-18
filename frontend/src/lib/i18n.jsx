import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'voice-agent-locale'

const translations = {
  en: {
    meta: { lang: 'en', dir: 'ltr', label: 'English' },
    nav: {
      brandTitle: 'Voice Agent',
      brandSubtitle: 'AI Call Intake Dashboard',
      call: 'Call',
      dashboard: 'Dashboard',
      switchLabel: 'AR',
    },
    call: {
      brandTitle: 'GB Voice Agent',
      brandSubtitle: 'Inbound AI calls in Egyptian Arabic',
      activeHint: 'Please speak clearly...',
      idleHint: 'Ready for an instant call',
      start: 'Start Call',
      end: 'End Call',
      connectingInline: 'Preparing the connection...',
      connectFallbackError:
        'Failed to connect. Make sure LiveKit env vars are configured correctly.',
      statusIdle: 'Ready for call',
      statusConnecting: 'Connecting...',
      statusInCall: 'Connected',
      statusEnded: 'Call ended',
      waveformActive: 'Tracking voice activity...',
      waveformIdle: 'Ready when you are',
    },
    dashboard: {
      title: 'Dashboard',
      subtitle:
        'Past calls are listed from newest to oldest with a quick inquiry summary.',
      search: 'Search',
      searchPlaceholder: 'Search by name or date...',
      loadError: 'Unable to load calls. Please try again later.',
      emptyTitle: 'No calls yet',
      emptySubtitle: 'New calls will appear here once they are completed.',
      unknownCaller: 'Unknown caller',
    },
    detail: {
      back: 'Back',
      loadError: 'Unable to load call details.',
      notFoundTitle: 'Call not found',
      notFoundSubtitle: 'Make sure the `callId` is correct and try again.',
      extractedTitle: 'Extracted record',
      extractedHint: 'Empty fields are shown as —',
      name: 'Name',
      address: 'Address',
      position: 'Position',
      inquiry: 'Inquiry',
      notes: 'Notes',
      transcript: 'Transcript',
      messages: 'messages',
      noTranscript: 'No transcript available yet.',
      caller: 'Caller',
      agent: 'Agent',
    },
  },
  ar: {
    meta: { lang: 'ar', dir: 'rtl', label: 'العربية' },
    nav: {
      brandTitle: 'Voice Agent',
      brandSubtitle: 'AI Call Intake Dashboard',
      call: 'مكالمة',
      dashboard: 'لوحة التحكم',
      switchLabel: 'EN',
    },
    call: {
      brandTitle: 'GB Voice Agent',
      brandSubtitle: 'استقبال مكالمات باللهجة المصرية',
      activeHint: 'الرجاء التحدث بوضوح…',
      idleHint: 'جاهز للاتصال الفوري',
      start: 'ابدأ المكالمة',
      end: 'إنهاء المكالمة',
      connectingInline: 'جاري تهيئة الاتصال…',
      connectFallbackError:
        'فشل الاتصال. تأكد من ضبط متغيرات LiveKit البيئية بشكل صحيح.',
      statusIdle: 'جاهز للمكالمة',
      statusConnecting: 'جار الاتصال...',
      statusInCall: 'متصل',
      statusEnded: 'تم إنهاء المكالمة',
      waveformActive: 'نشاط صوتي قيد التتبع...',
      waveformIdle: 'استعد للمكالمة',
    },
    dashboard: {
      title: 'لوحة التحكم',
      subtitle:
        'سجلات المكالمات السابقة مرتبة من الأحدث للأقدم، مع ملخص سريع للاستفسار.',
      search: 'بحث',
      searchPlaceholder: 'ابحث بالاسم أو التاريخ...',
      loadError: 'تعذر تحميل المكالمات. يرجى المحاولة لاحقًا.',
      emptyTitle: 'لا توجد مكالمات بعد',
      emptySubtitle: 'عندما تتم مكالمة جديدة، ستظهر هنا.',
      unknownCaller: 'متصل غير معروف',
    },
    detail: {
      back: 'رجوع',
      loadError: 'تعذر تحميل تفاصيل المكالمة.',
      notFoundTitle: 'لم يتم العثور على المكالمة',
      notFoundSubtitle: 'تأكد من أن `callId` صحيح ثم حاول مرة أخرى.',
      extractedTitle: 'البيانات المستخلصة',
      extractedHint: 'تظهر الحقول الفارغة كعلامة —',
      name: 'الاسم',
      address: 'العنوان',
      position: 'الوظيفة',
      inquiry: 'الاستفسار',
      notes: 'ملاحظات',
      transcript: 'المحادثة',
      messages: 'رسالة',
      noTranscript: 'لا توجد نسخة بعد.',
      caller: 'المتصل',
      agent: 'الوكيل',
    },
  },
}

const LocaleContext = createContext(null)

export function LocaleProvider({ children }) {
  const [locale, setLocale] = useState(() => {
    return localStorage.getItem(STORAGE_KEY) || 'en'
  })

  useEffect(() => {
    const current = translations[locale] || translations.en
    document.documentElement.lang = current.meta.lang
    document.documentElement.dir = current.meta.dir
    localStorage.setItem(STORAGE_KEY, locale)
  }, [locale])

  const value = useMemo(() => {
    const t = translations[locale] || translations.en
    return {
      locale,
      setLocale,
      toggleLocale: () => setLocale((prev) => (prev === 'en' ? 'ar' : 'en')),
      t,
      isArabic: t.meta.dir === 'rtl',
    }
  }, [locale])

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
}

export function useLocale() {
  const value = useContext(LocaleContext)
  if (!value) {
    throw new Error('useLocale must be used inside LocaleProvider')
  }
  return value
}


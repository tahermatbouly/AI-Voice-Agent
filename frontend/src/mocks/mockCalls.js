const now = Date.now()

function isoFromDaysAgo(daysAgo) {
  return new Date(now - daysAgo * 24 * 60 * 60 * 1000).toISOString()
}

export function getMockCalls() {
  return [
    {
      id: 'call_demo_1',
      callerName: 'أحمد محمد',
      inquirySummary: 'طلب صيانة لمكيف الهواء في الشقة.',
      timestamp: isoFromDaysAgo(1),
      name: 'أحمد محمد',
      address: '12 شارع النيل، القاهرة',
      position: 'محاسب',
      inquiry: 'المكيف بيعمل صوت عالي ومش بيبرد كويس.',
      notes: 'يرجى التواصل لتحديد موعد الزيارة.',
      transcript: [
        {
          role: 'caller',
          text: 'السلام عليكم، عندي مكيف في البيت بيعمل صوت وحرارة بتزيد.',
        },
        {
          role: 'agent',
          text: 'وعليكم السلام. تمام يا أحمد. ممكن أعرف العنوان بالظبط؟',
        },
        {
          role: 'caller',
          text: 'العنوان 12 شارع النيل في القاهرة.',
        },
        {
          role: 'agent',
          text: 'شكرًا. هل المكيف شغال لكن التبريد ضعيف، ولا مش بيشتغل خالص؟',
        },
        {
          role: 'caller',
          text: 'شغال بس التبريد ضعيف قوي.',
        },
        {
          role: 'agent',
          text: 'تمام. هنرتب فني يراجع الجهاز. هقولنا فين نتواصل معاك؟',
        },
        { role: 'caller', text: 'اتواصل على الرقم المسجل.' },
      ],
    },
    {
      id: 'call_demo_2',
      callerName: null,
      inquirySummary: 'استفسار عن فاتورة الخدمة الشهرية.',
      timestamp: isoFromDaysAgo(4),
      name: '',
      address: '',
      position: '',
      inquiry: 'عايز أعرف ليه الفاتورة زادت الشهر ده.',
      notes: 'المتصل لا يذكر بيانات شخصية.',
      transcript: [
        { role: 'caller', text: 'مرحبًا، الفاتورة الشهرية بقت أعلى من المعتاد.' },
        { role: 'agent', text: 'أكيد. هل تقدر تذكر رقم الحساب أو العداد إن وجد؟' },
        { role: 'caller', text: 'مش معايا دلوقتي، بس كنت محتاج أفهم السبب.' },
        {
          role: 'agent',
          text: 'تمام. هنراجع التفاصيل بعد ما توفر رقم الحساب، أو نطلب منك تزويد بياناتك لاحقًا.',
        },
      ],
    },
  ]
}

export function getMockCallById(id) {
  const calls = getMockCalls()
  return calls.find((c) => c.id === id) || null
}


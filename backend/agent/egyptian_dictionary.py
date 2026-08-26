"""
Fast Egyptian Arabic semantic dictionary for the recruitment voice agent.

IMPORTANT:
    This dictionary is LOCAL ONLY.

    Do NOT inject the entire dictionary into the LLM prompt.

Purpose:
    1. Normalize common Egyptian Arabic spellings.
    2. Detect common Egyptian words and phrases.
    3. Detect recruitment/HR-related intents.
    4. Reduce the amount of interpretation the LLM needs to perform.
    5. Provide deterministic signals to the extraction layer.

The dictionary intentionally maps Egyptian Arabic -> semantic meaning.
"""

# ============================================================
# VERSION
# ============================================================

DICTIONARY_VERSION = "1.0.0"


# ============================================================
# YES / NO
# ============================================================

YES = {
    "اه": "yes",
    "آه": "yes",
    "أه": "yes",
    "ايوه": "yes",
    "أيوه": "yes",
    "ايوة": "yes",
    "أيوة": "yes",
    "طبعا": "yes",
    "طبعاً": "yes",
    "أكيد": "yes",
    "اكيد": "yes",
    "حاضر": "yes",
    "تمام": "yes",
    "ماشي": "yes",
    "موافق": "yes",
    "موافقة": "yes",
}

NO = {
    "لا": "no",
    "لأ": "no",
    "لأه": "no",
    "لاء": "no",
    "مش": "no",
    "مش عايز": "does_not_want",
    "مش عايزة": "does_not_want",
    "مش حابب": "does_not_want",
    "مش حابة": "does_not_want",
    "مش مهتم": "not_interested",
    "مش مهتمة": "not_interested",
}


# ============================================================
# COMMON EGYPTIAN ARABIC
# ============================================================

COMMON = {

    # Pronouns
    "أنا": "i",
    "انا": "i",
    "إنت": "you",
    "انت": "you",
    "إنتي": "you",
    "انتي": "you",
    "هو": "he",
    "هي": "she",
    "إحنا": "we",
    "احنا": "we",
    "هم": "they",

    # Want
    "عايز": "want",
    "عاوز": "want",
    "عايزة": "want",
    "عاوزة": "want",

    # Need
    "محتاج": "need",
    "محتاجه": "need",
    "محتاجة": "need",

    # Have
    "عندي": "have",
    "عندك": "you_have",
    "عنده": "he_has",
    "عندها": "she_has",
    "عندنا": "we_have",
    "معايا": "i_have",
    "معاك": "you_have",
    "معاكي": "you_have",

    # Don't have
    "معنديش": "i_do_not_have",
    "ماعنديش": "i_do_not_have",
    "ما عنديش": "i_do_not_have",
    "معندكش": "you_do_not_have",
    "ماعندكش": "you_do_not_have",
    "معندوش": "he_does_not_have",
    "معندهاش": "she_does_not_have",

    # Existence
    "مفيش": "none",
    "مافيش": "none",
    "مفيش حاجة": "nothing",
    "فيه": "exists",
    "في": "exists_or_in",

    # Ability
    "أقدر": "can",
    "اقدر": "can",
    "مقدرش": "cannot",
    "ماقدرش": "cannot",
    "مش قادر": "cannot",
    "مش قادرة": "cannot",

    # Understanding
    "فاهم": "understand",
    "فاهمة": "understand",
    "فاهمك": "understand_you",
    "مش فاهم": "do_not_understand",
    "مش فاهمة": "do_not_understand",

    # Common
    "بس": "but_or_only",
    "كمان": "also",
    "برضه": "also",
    "برده": "also",
    "برضو": "also",
    "خلاص": "done_or_enough",
    "طيب": "transition",
    "طب": "transition",
    "كده": "like_this",
    "كدا": "like_this",
    "يعني": "filler_or_meaning",
    "يعنى": "filler_or_meaning",
    "شوية": "a_little",
    "معلش": "sorry_or_excuse_me",
    "لو سمحت": "please",
    "لو سمحتي": "please",
    "بص": "attention_marker",
    "بصي": "attention_marker",
    "بصراحة": "honestly",
    "على فكرة": "by_the_way",
    "بالمناسبة": "by_the_way",
}


# ============================================================
# TIME
# ============================================================

TIME = {

    "دلوقتي": "now",
    "دلوقت": "now",
    "دلوقتى": "now",
    "دلوقتِ": "now",

    "حالا": "now",
    "حالاً": "now",
    "حاليًا": "currently",
    "حاليا": "currently",

    "لسه": "still",
    "لسة": "still",
    "لسا": "still",
    "لسّا": "still",

    "النهارده": "today",
    "النهاردة": "today",
    "النهار دة": "today",

    "امبارح": "yesterday",
    "إمبارح": "yesterday",

    "بكرة": "tomorrow",
    "بكره": "tomorrow",
    "بُكرة": "tomorrow",

    "بعد بكرة": "day_after_tomorrow",

    "الأسبوع الجاي": "next_week",
    "الاسبوع الجاي": "next_week",

    "الأسبوع ده": "this_week",
    "الاسبوع ده": "this_week",

    "الأسبوع اللي جاي": "next_week",

    "الشهر الجاي": "next_month",

    "بعد شوية": "after_a_little_while",

    "قبل كده": "previously",
    "قبل كدا": "previously",
    "من قبل": "previously",

    "بعد كده": "later",
    "بعد كدا": "later",

    "فورًا": "immediately",
    "فورا": "immediately",
    "حالًا": "immediately",
}


# ============================================================
# RECRUITMENT / INTRODUCTION
# ============================================================

RECRUITMENT = {

    "مساء الخير": "greeting_evening",
    "صباح الخير": "greeting_morning",
    "أهلا بحضرتك": "greeting",
    "أهلاً بحضرتك": "greeting",
    "أهلا وسهلا": "greeting",
    "أهلاً وسهلاً": "greeting",

    "مع حضرتك": "this_is",
    "معاك": "speaking_with_you",
    "معاكي": "speaking_with_you",

    "مسؤول التوظيف": "recruitment_officer",
    "مسئول التوظيف": "recruitment_officer",
    "مسؤولة التوظيف": "recruitment_officer",

    "قسم التوظيف": "recruitment_department",
    "فريق التوظيف": "recruitment_team",

    "اتش ار": "human_resources",
    "إتش آر": "human_resources",
    "HR": "human_resources",

    "بكلم حضرتك": "calling_you",
    "بتصل بحضرتك": "calling_you",
    "باتصل بحضرتك": "calling_you",
    "بكلمك": "calling_you",

    "بخصوص": "regarding",
    "بخصوص فرص عمل": "regarding_job_opportunity",

    "فرصة عمل": "job_opportunity",
    "فرص عمل": "job_opportunities",
    "فرصة شغل": "job_opportunity",
    "فرص شغل": "job_opportunities",

    "وظيفة شاغرة": "vacancy",
    "وظائف شاغرة": "vacancies",
    "وظايف شاغرة": "vacancies",

    "وظيفة متاحة": "available_position",
    "وظائف متاحة": "available_positions",
    "وظايف متاحة": "available_positions",

    "فرص متاحة": "available_opportunities",

    "متاح حاليا": "currently_available",
    "متاحة حاليا": "currently_available",
    "متاح حاليًا": "currently_available",
    "متاحة حاليًا": "currently_available",

    "المكالمة مش هتاخد وقت": "call_will_not_take_long",
    "المكالمة مش هتاخد من وقت حضرتك": "call_will_not_take_long",
    "مش هتاخد غير دقيقتين": "two_minute_call",
    "دقيقتين تقريبا": "approximately_two_minutes",

    "نتأكد إن الوظيفة مناسبة": "confirm_job_suitability",
    "الوظيفة مناسبة لحضرتك": "job_is_suitable",

    "هل حضرتك مهتم": "asks_interest",
    "حضرتك مهتم": "asks_interest",
    "مهتم تسمع التفاصيل": "interested_in_details",
    "تحب تسمع التفاصيل": "interested_in_details",
    "تحب تعرف التفاصيل": "interested_in_details",
}


# ============================================================
# INTEREST / DECLINE
# ============================================================

INTEREST = {

    "اه مهتم": "interested",
    "آه مهتم": "interested",
    "أيوه مهتم": "interested",
    "ايوه مهتم": "interested",

    "اه مهتمة": "interested",
    "آه مهتمة": "interested",
    "أيوه مهتمة": "interested",

    "مهتم": "interested",
    "مهتمة": "interested",

    "عايز أعرف التفاصيل": "wants_details",
    "عايزة أعرف التفاصيل": "wants_details",
    "عايز اعرف التفاصيل": "wants_details",
    "عايزة اعرف التفاصيل": "wants_details",

    "آه ياريت": "yes_continue",
    "اه ياريت": "yes_continue",
    "ياريت": "yes_continue",

    "كمل": "continue",
    "كملي": "continue",
    "كمل التفاصيل": "continue",
    "كملي التفاصيل": "continue",
    "قول": "continue",
    "قولي": "continue",

    "مش مهتم": "not_interested",
    "مش مهتمة": "not_interested",

    "مش مناسب": "not_suitable",
    "مش مناسبة": "not_suitable",

    "مش عايز": "does_not_want",
    "مش عايزة": "does_not_want",

    "مش حابب": "does_not_want",
    "مش حابة": "does_not_want",

    "لا شكرا": "decline",
    "لا شكرًا": "decline",
    "متشكر": "thanks",
    "متشكرة": "thanks",
}


# ============================================================
# JOB / POSITION
# ============================================================

JOB = {

    "وظيفة": "job",
    "الوظيفة": "job",
    "وظيفه": "job",
    "الوظيفه": "job",

    "شغل": "job",
    "الشغل": "job",
    "شغلانة": "job",
    "شغلانه": "job",

    "فرصة": "opportunity",
    "فرصه": "opportunity",

    "البوزيشن": "position",
    "البوزشن": "position",
    "البوست": "position",

    "الوظيفة دي": "position",
    "الشغل ده": "position",
    "الشغل دا": "position",

    "تقديم": "application",
    "التقديم": "application",

    "أقدم": "apply",
    "اقدم": "apply",

    "تقدم": "apply",

    "أقدم على الوظيفة": "apply_for_job",
    "اقدم على الوظيفة": "apply_for_job",
    "أقدم على الشغل": "apply_for_job",
    "اقدم على الشغل": "apply_for_job",

    "حابب أقدم": "wants_to_apply",
    "حابة أقدم": "wants_to_apply",
    "حابب اقدم": "wants_to_apply",
    "حابة اقدم": "wants_to_apply",

    "عايز أقدم": "wants_to_apply",
    "عايزة أقدم": "wants_to_apply",
    "عايز اقدم": "wants_to_apply",
    "عايزة اقدم": "wants_to_apply",

    "فني ميكانيكا": "mechanical_technician",
    "فني ميكانيكا سيارات": "automotive_mechanic",
    "فني لحام": "welding_technician",
    "فني دهان": "painting_technician",
    "فني تشطيب": "finishing_technician",
    "فني مكينات": "machinery_technician",
    "فني تجليد": "grinding_or_finishing_technician",
    "فني تجليخ": "grinding_technician",
    "مساعد فني": "assistant_technician",
    "مساعد فني أقسام": "department_technician_assistant",

    "عامل": "worker",
    "عمال": "workers",
    "فني": "technician",
    "فنيين": "technicians",
    "مشرف": "supervisor",
    "مشرفين": "supervisors",
    "مهندس": "engineer",
    "مهندسين": "engineers",
}


# ============================================================
# EXPERIENCE
# ============================================================

EXPERIENCE = {

    "خبرة": "experience",
    "خبرتي": "my_experience",
    "خبرات": "experiences",

    "عندي خبرة": "has_experience",
    "معايا خبرة": "has_experience",
    "عندي خبرة سابقة": "has_previous_experience",
    "معايا خبرة سابقة": "has_previous_experience",

    "عندي خبرة في المجال": "has_field_experience",
    "معايا خبرة في المجال": "has_field_experience",

    "اشتغلت": "worked",
    "اشتغل": "work",
    "اشتغلت قبل كده": "worked_before",
    "اشتغلت قبل كدا": "worked_before",
    "اشتغلت من قبل": "worked_before",

    "اشتغلت في": "worked_at",
    "اشتغلت مع": "worked_with",

    "اشتغلت في المجال ده": "worked_in_field",
    "اشتغلت في نفس المجال": "worked_in_same_field",

    "كنت شغال": "was_working",
    "كنت شغالة": "was_working",
    "كنت بشتغل": "was_working",

    "بشتغل حاليا": "currently_working",
    "بشتغل حاليًا": "currently_working",
    "لسه شغال": "currently_working",
    "لسه شغالة": "currently_working",

    "سيبت الشغل": "left_job",
    "ساب الشغل": "left_job",

    "لسه متخرج": "recent_graduate",
    "لسه متخرجة": "recent_graduate",
    "حديث التخرج": "recent_graduate",
    "حديثة التخرج": "recent_graduate",

    "معنديش خبرة": "no_experience",
    "معنديش أي خبرة": "no_experience",
    "مفيش خبرة": "no_experience",
    "لسه معنديش خبرة": "no_experience",

    "سنين خبرة": "years_of_experience",
    "سنة خبرة": "years_of_experience",

    "اشتغلت في شركة": "worked_at_company",
    "اشتغلت في مصنع": "worked_at_factory",
}


# ============================================================
# EMPLOYER
# ============================================================

EMPLOYER = {

    "اشتغلت فين": "ask_previous_employer",
    "اشتغلت فين قبل كده": "ask_previous_employer",
    "اشتغلت فين قبل كدا": "ask_previous_employer",

    "اشتغلت في شركة ايه": "ask_previous_company",
    "اشتغلت في شركة إيه": "ask_previous_company",

    "اشتغلت فين بالظبط": "ask_previous_employer",
    "اشتغلت فين بالتحديد": "ask_previous_employer",

    "شركة ايه": "which_company",
    "شركة إيه": "which_company",

    "المكان اللي اشتغلت فيه": "previous_workplace",
    "مكان شغلك السابق": "previous_workplace",
    "شغلك السابق": "previous_job",

    "آخر شركة": "last_company",
    "آخر مكان اشتغلت فيه": "last_workplace",
    "الشركة السابقة": "previous_company",

    "اسم الشركة": "company_name",
    "اسم المصنع": "factory_name",

    "اشتغلت في السويدي": "worked_at_company",
}


# ============================================================
# SALARY
# ============================================================

SALARY = {

    "مرتب": "salary",
    "المرتب": "salary",
    "مرتبي": "my_salary",
    "المرتب بتاعي": "my_salary",

    "قبض": "salary",
    "القبض": "salary",
    "قبضي": "my_salary",

    "بقبض": "earn_salary",
    "باخد": "earn_salary",

    "بقبض كام": "ask_salary",
    "المرتب كام": "ask_salary",
    "المرتب كام يعني": "ask_salary",
    "القبض كام": "ask_salary",

    "بتقبض كام": "ask_current_salary",
    "كنت بتقبض كام": "ask_previous_salary",

    "أخبار المرتب إيه": "ask_salary",
    "اخبار المرتب ايه": "ask_salary",
    "المرتب أخباره إيه": "ask_salary",

    "المرتب بيبدأ من": "salary_starts_from",
    "المرتب يبدأ من": "salary_starts_from",
    "المرتب بيوصل": "salary_reaches",
    "المرتب لحد": "salary_up_to",

    "حسب الخبرة": "depends_on_experience",
    "حسب الخبرة والتقييم": "depends_on_experience_and_assessment",
    "حسب الخبرة والتقييم الفني": "depends_on_experience_and_technical_assessment",

    "التقييم الفني": "technical_assessment",
    "الاختبار الفني": "technical_test",
    "الاختبار": "test",

    "مرتب ثابت": "fixed_salary",
    "مرتب أساسي": "base_salary",
    "أساسي": "base_salary",

    "حوافز": "incentives",
    "حافز": "incentive",

    "بدل": "allowance",
    "بدل وجبة": "meal_allowance",
    "بدل أكل": "meal_allowance",
    "بدل مواصلات": "transportation_allowance",
    "بدل سكن": "housing_allowance",
}


# ============================================================
# WORK SCHEDULE
# ============================================================

WORK_SCHEDULE = {

    "نظام الشغل": "work_schedule",
    "نظام العمل": "work_schedule",
    "نظام الشغل ايه": "ask_work_schedule",
    "نظام الشغل إيه": "ask_work_schedule",
    "الشغل نظامه إيه": "ask_work_schedule",
    "الشغل نظامه ايه": "ask_work_schedule",

    "الشغل كام ساعة": "ask_working_hours",
    "كام ساعة شغل": "ask_working_hours",
    "عدد ساعات الشغل": "working_hours",
    "ساعات العمل": "working_hours",

    "8 ساعات": "eight_working_hours",
    "٨ ساعات": "eight_working_hours",
    "ثمان ساعات": "eight_working_hours",

    "ورديتين": "two_shifts",
    "وردية": "shift",
    "ورديات": "shifts",
    "نظام ورديات": "shift_system",
    "بنظام ورديتين": "two_shift_system",

    "شيفت": "shift",
    "شيفتات": "shifts",
    "شيفت صباحي": "morning_shift",
    "شيفت مسائي": "evening_shift",

    "وردية صباحية": "morning_shift",
    "وردية مسائية": "evening_shift",

    "الوردية بتتغير": "rotating_shifts",
    "الورديات متغيرة": "rotating_shifts",
    "شيفت متغير": "rotating_shift",

    "الجمعة والسبت": "friday_and_saturday",
    "إجازة الجمعة": "friday_off",
    "الإجازة الأسبوعية": "weekly_day_off",
    "الاجازة الأسبوعية": "weekly_day_off",
    "أجازة أسبوعية": "weekly_day_off",
    "إجازة اسبوعية": "weekly_day_off",

    "الويك إند": "weekend",
    "الويك اند": "weekend",
}


# ============================================================
# AVAILABILITY
# ============================================================

AVAILABILITY = {

    "متاح": "available",
    "متاحة": "available",
    "متوفر": "available",
    "متوفرة": "available",

    "أنا متاح": "available",
    "أنا متاحة": "available",

    "فاضي": "available",
    "فاضية": "available",
    "أنا فاضي": "available",
    "أنا فاضية": "available",

    "أقدر أبدأ": "can_start",
    "اقدر ابدأ": "can_start",

    "أقدر أبدأ دلوقتي": "can_start_now",
    "اقدر ابدأ دلوقتي": "can_start_now",

    "أقدر أبدأ بكرة": "can_start_tomorrow",
    "اقدر ابدأ بكرة": "can_start_tomorrow",

    "أقدر أبدأ الأسبوع الجاي": "can_start_next_week",
    "اقدر ابدأ الاسبوع الجاي": "can_start_next_week",

    "أقدر أبدأ فورًا": "can_start_immediately",
    "اقدر ابدأ فورا": "can_start_immediately",

    "ممكن أبدأ": "can_start",
    "ينفع أبدأ": "can_start",

    "مش متاح": "not_available",
    "مش متاحة": "not_available",
    "مش فاضي": "not_available",
    "مش فاضية": "not_available",

    "محتاج وقت": "needs_time",
    "محتاجة وقت": "needs_time",

    "محتاج أسبوع": "needs_one_week",
    "محتاج أسبوعين": "needs_two_weeks",

    "بعد أسبوع": "available_after_one_week",
    "بعد أسبوعين": "available_after_two_weeks",

    "عندي شغل حاليا": "currently_employed",
    "عندي شغل حاليًا": "currently_employed",
    "لسه شغال": "currently_employed",
    "لسه بشتغل": "currently_employed",

    "هسيب الشغل": "will_leave_current_job",
    "هسيب شغلي": "will_leave_current_job",
}


# ============================================================
# LOCATION
# ============================================================

LOCATION = {

    "مكان العمل": "work_location",
    "مكان الشغل": "work_location",
    "مكان الوظيفة": "work_location",

    "فين الشغل": "ask_work_location",
    "الشغل فين": "ask_work_location",
    "الوظيفة فين": "ask_work_location",
    "مكان العمل فين": "ask_work_location",

    "العين السخنة": "ain_sokhna",
    "عين السخنة": "ain_sokhna",

    "في المصنع": "at_factory",
    "في الشركة": "at_company",
    "داخل المصنع": "inside_factory",

    "مواصلات": "transportation",
    "في مواصلات": "transportation_available",
    "فيه مواصلات": "transportation_available",

    "الشركة بتوفر مواصلات": "company_provides_transportation",
    "مواصلات الشركة": "company_transportation",
    "اتوبيس الشركة": "company_bus",
    "أتوبيس الشركة": "company_bus",

    "سكن": "housing",
    "في سكن": "housing_available",
    "فيه سكن": "housing_available",
    "الشركة بتوفر سكن": "company_provides_housing",
}


# ============================================================
# BENEFITS
# ============================================================

BENEFITS = {

    "في وجبة": "meal_provided",
    "فيه وجبة": "meal_provided",
    "في وجبات": "meals_provided",
    "فيه وجبات": "meals_provided",

    "بيقدموا وجبة": "meal_provided",
    "بيقدموا أكل": "food_provided",

    "الشركة بتوفر وجبة": "company_provides_meal",
    "الشركة بتوفر وجبات": "company_provides_meals",

    "وجبة للعاملين": "employee_meal",
    "وجبة للعمال": "employee_meal",
    "وجبة للموظفين": "employee_meal",

    "بدل وجبة": "meal_allowance",
    "بدل أكل": "meal_allowance",

    "فيه بدل": "allowance_available",
    "في بدل": "allowance_available",

    "مميزات": "benefits",
    "المميزات": "benefits",
    "مميزات الوظيفة": "job_benefits",
    "مميزات الشغل": "job_benefits",

    "تأمين": "insurance",
    "في تأمين": "insurance_available",
    "فيه تأمين": "insurance_available",
    "تأمين طبي": "medical_insurance",
    "تأمين اجتماعي": "social_insurance",
}


# ============================================================
# EDUCATION
# ============================================================

EDUCATION = {

    "مؤهل": "qualification",
    "المؤهل": "qualification",
    "مؤهلك": "your_qualification",

    "مؤهل فني": "technical_qualification",
    "مؤهل صناعي": "industrial_qualification",

    "دبلوم صنايع": "industrial_diploma",
    "دبلوم صناعي": "industrial_diploma",

    "دبلوم صنايع ثلاث سنين": "three_year_industrial_diploma",
    "دبلوم خمس سنين": "five_year_technical_diploma",

    "ثانوي صنايع": "industrial_secondary_school",
    "ثانوية صناعية": "industrial_secondary_school",

    "مؤهل عالي": "higher_education",
    "مؤهل متوسط": "intermediate_education",
    "مؤهل فوق متوسط": "above_intermediate_education",

    "بكالوريوس": "bachelors_degree",
    "ليسانس": "bachelors_degree",
    "دبلوم": "diploma",

    "خريج": "graduate",
    "خريجة": "graduate",
    "متخرج": "graduate",
    "متخرجة": "graduate",

    "أنا خريج": "graduate",
    "أنا متخرج": "graduate",
}


# ============================================================
# AGE
# ============================================================

AGE = {

    "السن": "age",
    "سني": "my_age",
    "عندي كام سنة": "ask_age",
    "سن حضرتك كام": "ask_age",
    "عندك كام سنة": "ask_age",
    "حضرتك عندك كام سنة": "ask_age",

    "سنة": "year",
    "سنين": "years",
}


# ============================================================
# MILITARY SERVICE
# ============================================================

MILITARY = {

    "موقفك من التجنيد": "military_service_status",
    "موقف حضرتك من التجنيد": "military_service_status",
    "موقفك من الجيش": "military_service_status",
    "موقفك من الخدمة العسكرية": "military_service_status",

    "أديت الخدمة": "completed_military_service",
    "أديت الجيش": "completed_military_service",
    "أديت الخدمة العسكرية": "completed_military_service",
    "مأدي الخدمة": "completed_military_service",
    "مأدي الخدمة العسكرية": "completed_military_service",

    "خدمت الجيش": "completed_military_service",
    "خلصت الجيش": "completed_military_service",
    "خلصت الخدمة": "completed_military_service",

    "إعفاء": "exempt",
    "اعفاء": "exempt",
    "إعفاء نهائي": "permanent_exemption",
    "اعفاء نهائي": "permanent_exemption",
    "إعفاء مؤقت": "temporary_exemption",

    "تأجيل": "deferred",
    "تأجيل الخدمة": "military_service_deferred",

    "لسه مجند": "currently_serving",
    "لسه هقدم": "not_completed",
}


# ============================================================
# APPLICATION / INTERVIEW
# ============================================================

APPLICATION = {

    "أقدم إزاي": "how_to_apply",
    "اقدم ازاي": "how_to_apply",
    "أقدم ازاي يعني": "how_to_apply",
    "أعمل إيه عشان أقدم": "how_to_apply",

    "أقدم على الوظيفة إزاي": "how_to_apply",
    "أقدم على الشغل إزاي": "how_to_apply",

    "التقديم إزاي": "application_process",
    "طريقة التقديم": "application_process",

    "المقابلة إزاي": "interview_process",
    "المقابلة بتكون إزاي": "interview_process",
    "أعمل مقابلة إزاي": "interview_process",

    "المقابلة إمتى": "interview_date",
    "المقابلة امتى": "interview_date",
    "معاد المقابلة": "interview_date",
    "موعد المقابلة": "interview_date",

    "المقابلة فين": "interview_location",

    "هتحددوا معاد": "schedule_interview",
    "هيحددوا معاد": "schedule_interview",

    "هيتم التواصل": "will_be_contacted",
    "هيتم التواصل مع حضرتك": "will_be_contacted",
    "هتتواصلوا معايا": "will_contact_me",

    "الخطوة التالية": "next_step",
    "الخطوة الجاية": "next_step",
    "بعد التقديم": "after_application",
    "بعد المقابلة": "after_interview",

    "أبعت البيانات": "send_information",
    "أبعتلك البيانات": "send_information",

    "أبعت السيرة الذاتية": "send_cv",
    "أبعت السي في": "send_cv",

    "السي في": "cv",
    "السيرة الذاتية": "cv",
}


# ============================================================
# CANDIDATE QUESTIONS
# ============================================================

CANDIDATE_QUESTIONS = {

    "المرتب كام": "ask_salary",
    "المرتب إيه": "ask_salary",
    "المرتب ايه": "ask_salary",
    "المرتب أخباره إيه": "ask_salary",

    "الشغل كام ساعة": "ask_working_hours",
    "كام ساعة شغل": "ask_working_hours",

    "الشغل فين": "ask_work_location",
    "مكان الشغل فين": "ask_work_location",

    "في مواصلات": "ask_transportation",
    "فيه مواصلات": "ask_transportation",
    "المواصلات موجودة": "ask_transportation",

    "في وجبة": "ask_meal",
    "فيه وجبة": "ask_meal",
    "بيوفروا أكل": "ask_meal",

    "الإجازة إمتى": "ask_day_off",
    "الاجازة امتى": "ask_day_off",
    "الإجازة الأسبوعية إمتى": "ask_weekly_day_off",
    "الويك إند إمتى": "ask_weekly_day_off",

    "الشغل ورديات": "ask_shifts",
    "في ورديات": "ask_shifts",
    "فيه ورديات": "ask_shifts",
    "كام وردية": "ask_shift_count",

    "المقابلة فين": "ask_interview_location",
    "المقابلة إمتى": "ask_interview_date",

    "أقدم إزاي": "ask_application_process",
    "التقديم إزاي": "ask_application_process",

    "في سكن": "ask_housing",
    "فيه سكن": "ask_housing",

    "في تأمين": "ask_insurance",
    "في تأمين طبي": "ask_medical_insurance",

    "الوظيفة دي لسه متاحة": "ask_job_availability",
    "لسه التقديم مفتوح": "ask_application_status",

    "السن كام": "ask_age_requirement",
    "أقصى سن كام": "ask_maximum_age",
    "الحد الأقصى للسن": "ask_maximum_age",

    "المؤهل المطلوب إيه": "ask_required_qualification",
    "مطلوب مؤهل إيه": "ask_required_qualification",

    "مطلوب خبرة": "ask_experience_requirement",
    "كام سنة خبرة مطلوبة": "ask_experience_requirement",
}


# ============================================================
# CONVERSATION CONTROL
# ============================================================

CONVERSATION = {

    "استنى": "wait",
    "استني": "wait",
    "لحظة": "wait",
    "لحظه": "wait",

    "ممكن تعيد": "repeat_request",
    "ممكن تعيدي": "repeat_request",
    "ممكن تكرر": "repeat_request",
    "ممكن تكرري": "repeat_request",

    "مش سامع": "audio_not_heard",
    "مش سامعك": "audio_not_heard",
    "مش سامعاك": "audio_not_heard",

    "صوتك مش واضح": "audio_unclear",
    "مش واضح": "unclear",

    "معلش": "polite_interruption",

    "لحظة بس": "wait",
    "ثانية بس": "wait",
    "ثواني": "wait",

    "خلاص كده": "conversation_complete",
    "بس كده": "conversation_complete",

    "شكرا": "thanks",
    "شكرًا": "thanks",
    "شكرا ليك": "thanks",
    "شكرا لحضرتك": "thanks",

    "العفو": "you_are_welcome",
    "مع السلامة": "goodbye",
    "سلام": "goodbye",
}


# ============================================================
# EGYPTIAN NUMBERS
# ============================================================

NUMBERS = {

    "صفر": 0,

    "واحد": 1,
    "واحدة": 1,

    "اتنين": 2,
    "اثنين": 2,

    "تلاتة": 3,
    "ثلاثة": 3,

    "أربعة": 4,
    "اربعة": 4,

    "خمسة": 5,
    "ستة": 6,
    "سبعة": 7,

    "تمانية": 8,
    "تمنية": 8,
    "ثمانية": 8,

    "تسعة": 9,
    "عشرة": 10,

    "حداشر": 11,
    "اتناشر": 12,
    "تلاتاشر": 13,
    "أربعتاشر": 14,
    "خمستاشر": 15,
    "ستاشر": 16,
    "سبعتاشر": 17,
    "تمانتاشر": 18,
    "تمنتاشر": 18,
    "تسعتاشر": 19,

    "عشرين": 20,
    "تلاتين": 30,
    "ثلاثين": 30,
    "أربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "تمانين": 80,
    "تمنين": 80,
    "تسعين": 90,

    "مية": 100,
    "مائة": 100,

    "تلتمية": 300,
    "تلت مية": 300,

    "ألف": 1000,
    "الف": 1000,
}


# ============================================================
# SALARY EXPRESSIONS
# ============================================================

SALARY_VALUES = {

    "تمن تلاف": 8000,
    "تمانية آلاف": 8000,

    "تمن ونص": 8500,
    "تمانية ونص": 8500,

    "تسعة آلاف": 9000,
    "تسعة تلاف": 9000,

    "تسعة ونص": 9500,
    "تسعة ونص ألف": 9500,
    "تسعة آلاف ونص": 9500,

    "عشرة آلاف": 10000,
    "عشرة تلاف": 10000,
}


# ============================================================
# SPELLING NORMALIZATION
# ============================================================

SPELLING_VARIANTS = {

    # دلوقتي
    "دلوقت": "دلوقتي",
    "دلوقتى": "دلوقتي",
    "دلوقتِ": "دلوقتي",

    # tomorrow
    "بكره": "بكرة",
    "بُكرة": "بكرة",

    # how
    "ازاي": "إزاي",
    "ازاى": "إزاي",
    "إزاى": "إزاي",

    # yes
    "اه": "أه",
    "ايوه": "أيوه",
    "ايوة": "أيوة",

    # still
    "لسة": "لسه",
    "لسا": "لسه",
    "لسّا": "لسه",

    # want
    "عاوز": "عايز",
    "عايزة": "عايز",
    "عاوزة": "عايز",

    # because
    "عشان": "علشان",

    # yesterday
    "إمبارح": "امبارح",

    # today
    "النهاردة": "النهارده",
    "النهار دة": "النهارده",

    # what
    "ايه": "إيه",

    # when
    "امتى": "إمتى",

    # no
    "لاء": "لأ",

    # none
    "مافيش": "مفيش",

    # don't have
    "ماعنديش": "معنديش",
    "ما عنديش": "معنديش",
}


# ============================================================
# MERGED LOOKUP TABLE
# ============================================================

EGYPTIAN_DICTIONARY = {
    **YES,
    **NO,
    **COMMON,
    **TIME,
    **RECRUITMENT,
    **INTEREST,
    **JOB,
    **EXPERIENCE,
    **EMPLOYER,
    **SALARY,
    **WORK_SCHEDULE,
    **AVAILABILITY,
    **LOCATION,
    **BENEFITS,
    **EDUCATION,
    **AGE,
    **MILITARY,
    **APPLICATION,
    **CANDIDATE_QUESTIONS,
    **CONVERSATION,
}


# ============================================================
# FAST PHRASE-FIRST LOOKUP
# ============================================================

# Longer phrases should be checked before single words.
# This prevents:
#
#     "أقدر أبدأ بكرة"
#
# from being reduced to:
#
#     "can"
#
# before the more useful phrase is detected.

PHRASE_DICTIONARY = {}

for _dictionary in (
    RECRUITMENT,
    INTEREST,
    JOB,
    EXPERIENCE,
    EMPLOYER,
    SALARY,
    WORK_SCHEDULE,
    AVAILABILITY,
    LOCATION,
    BENEFITS,
    EDUCATION,
    MILITARY,
    APPLICATION,
    CANDIDATE_QUESTIONS,
    CONVERSATION,
):
    PHRASE_DICTIONARY.update(_dictionary)


SORTED_PHRASES = tuple(
    sorted(
        PHRASE_DICTIONARY.keys(),
        key=len,
        reverse=True,
    )
)


# ============================================================
# LOOKUP HELPERS
# ============================================================

def lookup_word(word: str):
    """
    O(1) lookup for a single Egyptian Arabic word.
    """
    return EGYPTIAN_DICTIONARY.get(word)


def normalize_word(word: str) -> str:
    """
    Normalize common STT spelling variations.
    """
    return SPELLING_VARIANTS.get(word, word)


def lookup_normalized_word(word: str):
    """
    Normalize first, then perform dictionary lookup.
    """
    normalized = normalize_word(word)
    return EGYPTIAN_DICTIONARY.get(normalized)


# ============================================================
# SEMANTIC CATEGORIES
# ============================================================

SEMANTIC_CATEGORIES = {

    "yes": {
        "yes",
    },

    "no": {
        "no",
        "does_not_want",
        "not_interested",
    },

    "experience": {
        "experience",
        "my_experience",
        "has_experience",
        "has_previous_experience",
        "has_field_experience",
        "worked",
        "worked_before",
        "worked_at",
        "worked_with",
        "worked_in_field",
        "worked_in_same_field",
        "currently_working",
        "no_experience",
        "years_of_experience",
    },

    "availability": {
        "available",
        "can_start",
        "can_start_now",
        "can_start_tomorrow",
        "can_start_next_week",
        "can_start_immediately",
        "not_available",
        "needs_time",
        "needs_one_week",
        "needs_two_weeks",
        "available_after_one_week",
        "available_after_two_weeks",
        "currently_employed",
    },

    "salary": {
        "salary",
        "my_salary",
        "earn_salary",
        "ask_salary",
        "ask_current_salary",
        "ask_previous_salary",
        "salary_starts_from",
        "salary_reaches",
        "salary_up_to",
        "depends_on_experience",
        "fixed_salary",
        "base_salary",
        "incentives",
        "allowance",
        "meal_allowance",
        "transportation_allowance",
        "housing_allowance",
    },

    "education": {
        "qualification",
        "your_qualification",
        "technical_qualification",
        "industrial_qualification",
        "industrial_diploma",
        "three_year_industrial_diploma",
        "five_year_technical_diploma",
        "industrial_secondary_school",
        "higher_education",
        "intermediate_education",
        "above_intermediate_education",
        "bachelors_degree",
        "diploma",
        "graduate",
    },

    "military": {
        "military_service_status",
        "completed_military_service",
        "exempt",
        "permanent_exemption",
        "temporary_exemption",
        "deferred",
        "military_service_deferred",
        "currently_serving",
        "not_completed",
    },

    "job": {
        "job",
        "position",
        "opportunity",
        "job_opportunity",
        "job_opportunities",
        "available_position",
        "available_positions",
        "apply",
        "apply_for_job",
        "wants_to_apply",
    },

    "location": {
        "work_location",
        "ask_work_location",
        "ain_sokhna",
        "transportation",
        "transportation_available",
        "company_provides_transportation",
        "housing",
        "housing_available",
    },

    "benefits": {
        "meal_provided",
        "meals_provided",
        "food_provided",
        "company_provides_meal",
        "company_provides_meals",
        "meal_allowance",
        "allowance_available",
        "benefits",
        "job_benefits",
        "insurance",
        "insurance_available",
        "medical_insurance",
        "social_insurance",
    },

    "application": {
        "application",
        "how_to_apply",
        "application_process",
        "interview_process",
        "interview_date",
        "interview_location",
        "schedule_interview",
        "will_be_contacted",
        "next_step",
        "after_application",
        "after_interview",
        "send_information",
        "send_cv",
    },
}


# ============================================================
# HR PRIORITY TERMS
# ============================================================

# These are the terms that matter most to your current agent.
#
# You can use this set to make the local semantic layer focus
# on the information required by your extraction workflow.

HIGH_PRIORITY_TERMS = {
    "experience",
    "has_experience",
    "has_previous_experience",
    "worked_before",
    "years_of_experience",

    "available",
    "can_start",
    "can_start_now",
    "can_start_tomorrow",
    "can_start_next_week",
    "not_available",

    "job",
    "position",
    "wants_to_apply",

    "salary",
    "ask_salary",

    "qualification",
    "technical_qualification",
    "industrial_diploma",

    "military_service_status",
    "completed_military_service",
    "exempt",
    "deferred",

    "work_location",
    "working_hours",
    "shift",
    "two_shifts",

    "meal_provided",
    "meal_allowance",

    "transportation",
    "housing",

    "interview_process",
    "application_process",
}
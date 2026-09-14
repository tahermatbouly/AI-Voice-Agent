import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from Backend import config
from Backend.agent.state import AgentState


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0.2,
)


# ============================================================
# Known STT artifacts / obviously invalid answers
# ============================================================

INVALID_TRANSCRIPTS = {
    "",
    "لا",
    "لأ",
    "مش عارف",
    "مش عارفة",
    "معرفش",
    "ماعرفش",
    "ونكمل",
    "منور",
    "تمام",
    "نعم",
    "اه",
    "آه",
    "أيوه",
    "@@فراغ",
    "فراغ",
}


def is_obviously_invalid(transcript: str) -> bool:

    normalized = (
        transcript
        .strip()
        .lower()
    )

    return normalized in INVALID_TRANSCRIPTS


# ============================================================
# Dynamic tool creation — accept the current question's answer
# ============================================================

def create_update_tool(
    target_fields: list[str],
):
    """
    Create a tool that can ONLY update fields belonging
    to the current interview question.
    """

    fields = {}

    for field in target_fields:
        fields[field] = (
            str | None,
            Field(
                default=None,
                description=(
                    f"Value extracted for {field}. "
                    "Only provide this if the candidate "
                    "actually answered the current question."
                ),
            ),
        )

    UpdateModel = create_model(
        "CandidateUpdate",
        **fields,
    )

    def update_candidate_info(**kwargs):

        updates = {}

        for field, value in kwargs.items():

            if value is None:
                continue

            if isinstance(value, str):
                value = value.strip()

                if not value:
                    continue

            updates[field] = value

        return updates

    return StructuredTool.from_function(
        func=update_candidate_info,
        name="update_candidate_info",
        description=(
            "Update candidate information ONLY when the "
            "candidate has actually answered the current "
            "interview question. "
            "DO NOT call this tool for greetings, yes/no "
            "acknowledgements, refusals, unrelated answers, "
            "questions, or unclear speech."
        ),
        args_schema=UpdateModel,
    )


# ============================================================
# Extraction prompt
# ============================================================

EXTRACTION_PROMPT = """
You are a STRICT answer validation and information
extraction system for an Egyptian Arabic recruitment interview.

Your job is simple:

1. Look at the CURRENT QUESTION.
2. Look at the CANDIDATE ANSWER.
3. Decide whether the candidate actually answered
   the current question.
4. If they answered it, call update_candidate_info.
5. If they did NOT answer it, DO NOT call the tool.

==================================================
VERY IMPORTANT
==================================================

A tool call means:

"THE CANDIDATE ANSWERED THE CURRENT QUESTION."

Therefore NEVER call the tool just because the candidate
said something understandable.

The answer MUST contain information relevant to the
CURRENT QUESTION.

==================================================
INVALID ANSWERS
==================================================

Do NOT call the tool for:

"لا"
"لأ"
"مش عارف"
"مش عارفة"
"معرفش"
"ماعرفش"
"نعم"
"آه"
"أيوه"
"تمام"
"منور"
"ونكمل"
"فراغ"
"@@فراغ"

Also do NOT call the tool for:

- greetings
- acknowledgements
- confirmations
- questions
- unrelated statements
- unclear speech
- meaningless speech
- an answer to another interview question

==================================================
EXAMPLES
==================================================

CURRENT QUESTION:
ممكن أعرف اسم حضرتك بالكامل؟

CANDIDATE:
إي نعم

RESULT:
DO NOT CALL THE TOOL.

--------------------------------------------------

CURRENT QUESTION:
ممكن أعرف اسم حضرتك بالكامل؟

CANDIDATE:
أنا طاهر محمد

RESULT:
CALL update_candidate_info with:

candidate_name = "طاهر محمد"

--------------------------------------------------

CURRENT QUESTION:
إيه المجال أو الشغلانة اللي بتدور عليها؟

CANDIDATE:
@@فراغ

RESULT:
DO NOT CALL THE TOOL.

--------------------------------------------------

CURRENT QUESTION:
إيه المجال أو الشغلانة اللي بتدور عليها؟

CANDIDATE:
أنا بدور على شغل في مجال الذكاء الاصطناعي

RESULT:
CALL update_candidate_info with:

target_domain = "الذكاء الاصطناعي"

--------------------------------------------------

CURRENT QUESTION:
عندك كام سنة خبرة في المجال ده؟

CANDIDATE:
ونكمل

RESULT:
DO NOT CALL THE TOOL.

--------------------------------------------------

CURRENT QUESTION:
عندك كام سنة خبرة في المجال ده؟

CANDIDATE:
عندي 3 سنين خبرة

RESULT:
CALL update_candidate_info with:

years_of_experience = "3"

--------------------------------------------------

CURRENT QUESTION:
عندك كام سنة خبرة في المجال ده؟

CANDIDATE:
لسه متخرج وممعنديش خبرة

RESULT:
CALL update_candidate_info with:

years_of_experience = "0"

ZERO EXPERIENCE IS A VALID ANSWER.

--------------------------------------------------

CURRENT QUESTION:
إيه مؤهلك الدراسي أو أعلى شهادة معاك؟

CANDIDATE:
إنه مش فاهم معي

RESULT:
DO NOT CALL THE TOOL.

--------------------------------------------------

CURRENT QUESTION:
إيه مؤهلك الدراسي أو أعلى شهادة معاك؟

CANDIDATE:
أنا بدرس علوم حاسب

RESULT:
CALL update_candidate_info with:

education_level = "علوم حاسب"

--------------------------------------------------

CURRENT QUESTION:
إيه أهم المهارات والبرامج أو الأدوات اللي بتستخدمها باستمرار؟

CANDIDATE:
نعم

RESULT:
DO NOT CALL THE TOOL.

--------------------------------------------------

CURRENT QUESTION:
إيه أهم المهارات والبرامج أو الأدوات اللي بتستخدمها باستمرار؟

CANDIDATE:
Python و SQL وبستخدم Git و Docker

RESULT:
CALL update_candidate_info with:

key_skills = ["Python", "SQL"]
tools_technologies = ["Git", "Docker"]

==================================================
CRITICAL RULE
==================================================

Do not extract information from an answer unless the answer
actually answers the CURRENT QUESTION.

For example:

Question:
ممكن أعرف اسم حضرتك بالكامل؟

Answer:
عندي 3 سنين خبرة.

DO NOT call the tool.

Even though "3 سنين خبرة" is useful candidate information,
it answers a different question.

==================================================
CURRENT QUESTION
==================================================

{question}

==================================================
ALLOWED FIELDS
==================================================

{target_fields}

==================================================
CANDIDATE ANSWER
==================================================

{transcript}
"""


# ============================================================
# Extraction
# ============================================================

async def extract_candidate_info(
    state: AgentState,
) -> AgentState:

    question = state.get("current_question")

    if not question:

        print(
            "[EXTRACTION] No current question."
        )

        return {
            **state,
            "extraction_success": False,
            "failure_reason": "no_speech",
        }

    transcript = (
        state.get("transcript", "")
        .strip()
    )

    # --------------------------------------------------------
    # Empty / obvious STT artifact — skip the LLM call entirely,
    # we already know this is the "unclear" case.
    # --------------------------------------------------------

    if is_obviously_invalid(transcript):

        print(
            "[EXTRACTION] "
            f"Rejected obvious invalid transcript: "
            f"{transcript!r}"
        )

        return {
            **state,
            "extraction_success": False,
            "failure_reason": "no_speech",
        }

    # --------------------------------------------------------
    # Determine allowed fields
    # --------------------------------------------------------

    if "target_field" in question:

        target_fields = [
            question["target_field"]
        ]

    elif "target_fields" in question:

        target_fields = question["target_fields"]

    else:

        print(
            "[EXTRACTION] "
            "Question has no target fields."
        )

        return {
            **state,
            "extraction_success": False,
            "failure_reason": "no_speech",
        }

    print(
        "[EXTRACTION] Question:",
        question["id"],
    )

    print(
        "[EXTRACTION] Transcript:",
        transcript,
    )

    print(
        "[EXTRACTION] Allowed fields:",
        target_fields,
    )

    # --------------------------------------------------------
    # Create question-specific tool, bound alongside the static
    # reject_answer tool so the LLM must always pick one or the
    # other — never call nothing.
    # --------------------------------------------------------

    update_tool = create_update_tool(
        target_fields
    )

    extraction_llm = llm.bind_tools(
        [update_tool]
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = EXTRACTION_PROMPT.format(
        question=question["question_ar"],
        target_fields=", ".join(target_fields),
        transcript=transcript,
    )

    messages = [
        SystemMessage(
            content=prompt
        ),
        HumanMessage(
            content=transcript
        ),
    ]

    # --------------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------------

    response = await extraction_llm.ainvoke(
        messages
    )

    print(
        "[EXTRACTION] Raw response:",
        response.content,
    )

    print(
        "[EXTRACTION] Tool calls:",
        response.tool_calls,
    )

    # --------------------------------------------------------
    # No tool call = the model didn't extract anything for this
    # question. Since is_obviously_invalid() already caught genuine
    # empty/filler/silence transcripts earlier and returned before
    # we ever got here, whatever reached this point had real,
    # transcribed speech in it — so by elimination this means "said
    # something, just not an answer to this question", not
    # "didn't hear anything".
    # --------------------------------------------------------

    if not response.tool_calls:

        print(
            "[EXTRACTION] "
            "No tool call -> answer rejected."
        )

        return {
            **state,
            "extraction_success": False,
            "failure_reason": "wrong_answer",
        }

    # --------------------------------------------------------
    # Process update_candidate_info call
    # --------------------------------------------------------

    candidate = {
        **state["candidate"]
    }

    extracted_any = False

    for tool_call in response.tool_calls:

        if (
            tool_call["name"]
            != "update_candidate_info"
        ):
            continue

        args = tool_call.get(
            "args",
            {}
        )

        print(
            "[EXTRACTION] Tool arguments:",
            args,
        )

        for field in target_fields:

            value = args.get(field)

            if value is None:
                continue

            if isinstance(value, str):

                value = value.strip()

                if not value:
                    continue

            candidate[field] = value

            extracted_any = True

    # --------------------------------------------------------
    # Tool call without useful data
    # --------------------------------------------------------

    if not extracted_any:

        print(
            "[EXTRACTION] "
            "Tool was called but no useful "
            "information was provided."
        )

        return {
            **state,
            "extraction_success": False,
            "failure_reason": "wrong_answer",
        }

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "[EXTRACTION] "
        "Answer accepted."
    )

    print(
        "[EXTRACTION] Updated candidate:",
        candidate,
    )

    return {
        **state,
        "candidate": candidate,
        "extraction_success": True,
        "failure_reason": None,
    }


# ============================================================
# Retry / clarification lines
# ============================================================
# Spoken before repeating a question after a rejected answer. The
# actual Arabic prefix text lives in questions.json (as the
# "no_speech_retry" / "wrong_answer_retry" system_message entries)
# rather than here — this just joins whatever prefix graph.py looks
# up with that question's own question_ar.

def build_retry_text(
    question_ar: str,
    prefix: str,
) -> str:

    return f"{prefix}{question_ar}"


# ============================================================
# End-of-interview summary
# ============================================================
# Everything below supports the "review your answers" step: turning
# the collected candidate dict into spoken Arabic text, and later
# classifying whether the candidate's reply to that summary means
# "confirm, we're done" or "change one specific field".

FIELD_LABELS: dict[str, str] = {
    "candidate_name": "الاسم",
    "target_domain": "المجال أو الشغلانة",
    "years_of_experience": "سنين الخبرة",
    "education_level": "المؤهل الدراسي",
    "key_skills": "المهارات",
    "tools_technologies": "الأدوات والبرامج",
    "english_proficiency": "مستوى الإنجليزي",
}

# Fixed, readable order for the spoken summary — independent of
# whatever order fields happened to be extracted/updated in.
_SUMMARY_FIELD_ORDER = [
    "candidate_name",
    "target_domain",
    "years_of_experience",
    "education_level",
    "key_skills",
    "tools_technologies",
    "english_proficiency",
]


def _format_summary_value(value) -> str:

    if isinstance(value, list):
        return "، ".join(str(item) for item in value)

    return str(value)


def build_summary_text(candidate: dict) -> str:
    """
    Builds the Arabic text spoken back to the candidate at the end
    of the interview, listing every field collected so far. Called
    both the first time (after the last question) and again after
    every correction, since the values change each time.
    """

    lines = []

    for field in _SUMMARY_FIELD_ORDER:

        if field not in candidate:
            continue

        label = FIELD_LABELS.get(field, field)

        lines.append(
            f"{label}: {_format_summary_value(candidate[field])}"
        )

    body = "، ".join(lines)

    return (
        "خليني أراجع مع حضرتك البيانات اللي سجلتها: "
        f"{body}. "
        "لو كله تمام قول تمام أو صح عشان نخلص المكالمة، "
        "ولو حابب تعدل حاجة قول مثلاً غيرلي المؤهل أو غيرلي الخبرة."
    )


# ------------------------------------------------------------
# Summary reply classification tool
# ------------------------------------------------------------

class SummaryReplyArgs(BaseModel):

    action: str = Field(
        ...,
        description=(
            "'confirm' if the candidate is happy with the summary "
            "and wants to finish the call. 'correct' if they want "
            "to change one specific piece of information."
        ),
    )

    field: str | None = Field(
        default=None,
        description=(
            "Required when action is 'correct'. Must be exactly "
            "one of: candidate_name, target_domain, "
            "years_of_experience, education_level, key_skills, "
            "tools_technologies, english_proficiency. "
            "Omit when action is 'confirm'."
        ),
    )


def _resolve_summary_reply(
    action: str,
    field: str | None = None,
):
    return {
        "action": action,
        "field": field,
    }


resolve_summary_reply_tool = StructuredTool.from_function(
    func=_resolve_summary_reply,
    name="resolve_summary_reply",
    description=(
        "Call this once you have decided whether the candidate "
        "confirmed the summary as correct, or asked to change one "
        "specific field in it."
    ),
    args_schema=SummaryReplyArgs,
)

summary_reply_llm = llm.bind_tools(
    [resolve_summary_reply_tool]
)


SUMMARY_REPLY_PROMPT = """
You are classifying an Egyptian Arabic candidate's reply to a
spoken summary of their interview answers.

Decide exactly one of two things:

1. The candidate CONFIRMED the summary is correct and wants to
   finish the call.
2. The candidate wants to CORRECT one specific field.

Always call resolve_summary_reply with your decision.

==================================================
VALID FIELDS FOR CORRECTION
==================================================

candidate_name          — الاسم
target_domain           — المجال أو الشغلانة
years_of_experience     — سنين الخبرة
education_level         — المؤهل الدراسي
key_skills              — المهارات
tools_technologies      — الأدوات والبرامج
english_proficiency     — مستوى الإنجليزي

==================================================
EXAMPLES
==================================================

CANDIDATE:
تمام و صح و شكرا

RESULT:
action = "confirm"

--------------------------------------------------

CANDIDATE:
ايوة كده تمام خلاص

RESULT:
action = "confirm"

--------------------------------------------------

CANDIDATE:
غيرلي المؤهل

RESULT:
action = "correct"
field = "education_level"

--------------------------------------------------

CANDIDATE:
لأ عايز أعدل سنين الخبرة

RESULT:
action = "correct"
field = "years_of_experience"

--------------------------------------------------

CANDIDATE:
ممكن تغير الاسم، اسمي غلط

RESULT:
action = "correct"
field = "candidate_name"

--------------------------------------------------

CANDIDATE:
غيرلي المهارات والأدوات

RESULT:
action = "correct"
field = "key_skills"

==================================================
CANDIDATE REPLY
==================================================

{transcript}
"""


async def classify_summary_reply(
    transcript: str,
) -> dict:
    """
    Returns one of:
        {"action": "confirm", "field": None}
        {"action": "correct", "field": <one of FIELD_LABELS>}
        {"action": "unclear", "field": None}

    "unclear" covers both a genuinely ambiguous reply and a
    'correct' request naming a field we don't recognize — in both
    cases the caller should just replay the summary rather than
    guess.
    """

    transcript = transcript.strip()

    if not transcript:
        return {"action": "unclear", "field": None}

    prompt = SUMMARY_REPLY_PROMPT.format(
        transcript=transcript,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=transcript),
    ]

    response = await summary_reply_llm.ainvoke(
        messages
    )

    print(
        "[SUMMARY] Raw response:",
        response.content,
    )

    print(
        "[SUMMARY] Tool calls:",
        response.tool_calls,
    )

    if not response.tool_calls:
        return {"action": "unclear", "field": None}

    args = response.tool_calls[0].get("args", {})

    action = args.get("action")
    field = args.get("field")

    if action == "confirm":
        return {"action": "confirm", "field": None}

    if action == "correct" and field in FIELD_LABELS:
        return {"action": "correct", "field": field}

    print(
        "[SUMMARY] Unrecognized action/field:",
        action,
        field,
    )

    return {"action": "unclear", "field": None}
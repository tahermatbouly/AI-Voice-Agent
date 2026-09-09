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
# Dynamic tool creation
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
        }

    transcript = (
        state.get("transcript", "")
        .strip()
    )

    # --------------------------------------------------------
    # Empty / obvious STT artifact
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
    # Create question-specific tool
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
    # No tool call = invalid answer
    # --------------------------------------------------------

    if not response.tool_calls:

        print(
            "[EXTRACTION] "
            "No tool call -> answer rejected."
        )

        return {
            **state,
            "extraction_success": False,
        }

    # --------------------------------------------------------
    # Process tool call
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
    }
from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import InterviewManager
from Backend.agent.extraction import (
    extract_candidate_info,
    build_summary_text,
    classify_summary_reply,
    build_retry_text,
)


QUESTIONS_PATH = "Backend/agent/questions.json"

interview_manager = InterviewManager(QUESTIONS_PATH)


# ============================================================
# Initialize interview
# ============================================================

async def initialize_interview(
    state: AgentState,
) -> AgentState:

    welcome = interview_manager.get_question(0)

    print(
        "[GRAPH] Initializing interview"
    )

    print(
        "[GRAPH] Welcome question:",
        welcome["id"] if welcome else None,
    )

    return {
        **state,
        "questions": interview_manager.questions,
        "current_question_index": 0,
        "current_question": welcome,
        "response": (
            welcome["question_ar"]
            if welcome
            else ""
        ),
        "interview_finished": False,
        "extraction_success": False,
        "transcript": "",
        "mode": "interview",
        "failure_reason": None,
    }


# ============================================================
# Extraction
# ============================================================

async def extract_answer(
    state: AgentState,
) -> AgentState:

    print(
        "[GRAPH] Extracting answer for:",
        (
            state["current_question"]["id"]
            if state.get("current_question")
            else None
        ),
    )

    return await extract_candidate_info(state)


# ============================================================
# Question lookup helpers
# ============================================================

SUMMARY_QUESTION_ID = "summary"

# ids of the two clarification lines stored as system_message
# entries at the end of questions.json.
NO_SPEECH_RETRY_ID = "no_speech_retry"
WRONG_ANSWER_RETRY_ID = "wrong_answer_retry"

# Used only if the corresponding entry is ever missing from
# questions.json — keeps repeat_question from breaking outright.
_FALLBACK_PREFIXES = {
    "no_speech": "معلش، مسمعتش حضرتك كويس. ",
    "wrong_answer": "معلش، الإجابة دي مش بتجاوب على السؤال ده. ",
}

_RETRY_SUFFIXES = (
    "__retry_no_speech",
    "__retry_wrong_answer",
)


def _base_question_id(question_id: str) -> str:
    """
    A repeated question's id gets suffixed (see repeat_question
    below) so speak() doesn't play stale cached audio for the wrong
    text. This strips that suffix back off so the pristine original
    question can always be found again, however many times in a row
    it gets repeated — without this, a second consecutive failure
    would re-prefix already-prefixed text instead of starting fresh.
    """

    for suffix in _RETRY_SUFFIXES:

        if question_id.endswith(suffix):
            return question_id[: -len(suffix)]

    return question_id


def _find_question_by_id(question_id: str) -> dict | None:

    for question in interview_manager.questions:

        if question.get("id") == question_id:
            return question

    return None


def _retry_prefix_text(failure_reason: str) -> str:
    """
    Looks up the clarification line's text from its system_message
    entry in questions.json, falling back to a hardcoded copy only
    if that entry is somehow missing.
    """

    prefix_id = (
        WRONG_ANSWER_RETRY_ID
        if failure_reason == "wrong_answer"
        else NO_SPEECH_RETRY_ID
    )

    prefix_question = _find_question_by_id(prefix_id)

    if prefix_question is not None:
        return prefix_question["question_ar"]

    print(
        "[GRAPH] WARNING: system_message"
        f" '{prefix_id}' missing from questions.json,"
        " using fallback text",
    )

    return _FALLBACK_PREFIXES[
        failure_reason
        if failure_reason in _FALLBACK_PREFIXES
        else "no_speech"
    ]


def _find_question_by_field(field: str | None) -> dict | None:
    """
    Maps a candidate field name (e.g. "education_level") back to the
    interview question that collects it (e.g. q4_education), so a
    correction request can jump straight there.
    """

    if not field:
        return None

    for question in interview_manager.questions:

        if question.get("target_field") == field:
            return question

        if field in question.get("target_fields", []):
            return question

    return None


# ============================================================
# Summary
# ============================================================

async def build_summary(
    state: AgentState,
) -> AgentState:
    """
    Builds the spoken end-of-interview summary from whatever's in
    state["candidate"] right now. Reused both the first time (after
    the last real question) and again after every correction, since
    the values change each time.

    Deliberately returns extraction_success=True: from handler.py's
    point of view this is indistinguishable from "here is the next
    question to ask" — it reuses that exact code path, so the
    summary just gets spoken like any other question, with no
    changes needed in handler.py.
    """

    candidate = state.get("candidate", {})

    summary_text = build_summary_text(candidate)

    summary_question = {
        "id": SUMMARY_QUESTION_ID,
        "question_ar": summary_text,
    }

    print(
        "[GRAPH] Built summary for confirmation"
    )

    return {
        **state,
        "current_question": summary_question,
        "response": summary_text,
        "mode": "summary",
        "transcript": "",
        "extraction_success": True,
        "interview_finished": False,
        "failure_reason": None,
    }


async def classify_summary(
    state: AgentState,
) -> AgentState:
    """
    Handles the candidate's reply to the summary: either they
    confirmed it (=> finish), asked to change a specific field
    (=> jump back to that question), or said something ambiguous
    (=> replay the summary rather than guess).
    """

    transcript = state.get("transcript", "").strip()

    decision = await classify_summary_reply(transcript)

    print(
        "[GRAPH] Summary reply decision:",
        decision,
    )

    # --------------------------------------------------------
    # Confirmed — finish the interview. This reuses the exact
    # same (extraction_success=True, interview_finished=True)
    # shape advance_question used to produce, so handler.py's
    # existing "interview finished" branch handles it unchanged.
    # --------------------------------------------------------

    if decision["action"] == "confirm":

        return {
            **state,
            "extraction_success": True,
            "interview_finished": True,
            "transcript": "",
            "failure_reason": None,
        }

    # --------------------------------------------------------
    # Correction requested — jump back to the matching question.
    # --------------------------------------------------------

    if decision["action"] == "correct":

        target_question = _find_question_by_field(
            decision.get("field")
        )

        if target_question is not None:

            print(
                "[GRAPH] Jumping back to:",
                target_question["id"],
            )

            return {
                **state,
                "current_question": target_question,
                "response": target_question["question_ar"],
                "mode": "correcting",
                "extraction_success": True,
                "interview_finished": False,
                "transcript": "",
                "failure_reason": None,
            }

    # --------------------------------------------------------
    # Unclear reply, or a field we couldn't map to a question —
    # replay the summary rather than guess what they meant. Mode
    # stays "summary" (untouched via the state spread below) so
    # repeat_question knows this isn't a normal question retry.
    # --------------------------------------------------------

    print(
        "[GRAPH] Summary reply unclear, replaying summary"
    )

    return {
        **state,
        "extraction_success": False,
        "transcript": "",
        "failure_reason": None,
    }


# ============================================================
# START routing
# ============================================================

def route_from_start(
    state: AgentState,
):

    current_question = state.get(
        "current_question"
    )

    transcript = state.get(
        "transcript",
        "",
    ).strip()

    mode = state.get(
        "mode",
        "interview",
    )

    print(
        "[GRAPH] START ROUTER"
    )

    print(
        "[GRAPH] Current question:",
        (
            current_question["id"]
            if current_question
            else None
        ),
    )

    print(
        "[GRAPH] Transcript:",
        transcript,
    )

    print(
        "[GRAPH] Mode:",
        mode,
    )

    # --------------------------------------------------------
    # First invocation
    # --------------------------------------------------------

    if current_question is None:

        print(
            "[GRAPH] Route -> initialize"
        )

        return "initialize"

    # --------------------------------------------------------
    # We already have a question.
    # Therefore this invocation is processing
    # the candidate's answer.
    # --------------------------------------------------------

    if transcript:

        if mode == "summary":

            print(
                "[GRAPH] Route -> classify_summary"
            )

            return "classify_summary"

        print(
            "[GRAPH] Route -> extract"
        )

        return "extract"

    # --------------------------------------------------------
    # No question and no transcript
    # --------------------------------------------------------

    print(
        "[GRAPH] Route -> initialize"
    )

    return "initialize"


# ============================================================
# After extraction
# ============================================================

def route_after_extraction(
    state: AgentState,
):

    success = state.get(
        "extraction_success",
        False,
    )

    mode = state.get(
        "mode",
        "interview",
    )

    print(
        "[GRAPH] Extraction success:",
        success,
    )

    if not success:

        print(
            "[GRAPH] Route -> repeat"
        )

        return "repeat"

    # A successful answer while correcting a summary field goes
    # back to the summary (rebuilt with the new value), not forward
    # through the normal question sequence.
    if mode == "correcting":

        print(
            "[GRAPH] Route -> summarize (post-correction)"
        )

        return "summarize"

    print(
        "[GRAPH] Route -> advance"
    )

    return "advance"


# ============================================================
# After advancing
# ============================================================

def route_after_advance(
    state: AgentState,
):
    """
    advance_question sets current_question to None once it walks
    past the last entry in questions.json. That's the signal to
    build the summary instead of ending the graph normally.
    """

    if state.get("current_question") is None:

        print(
            "[GRAPH] Route -> summarize (end of questions)"
        )

        return "summarize"

    return "end"


# ============================================================
# After summary classification
# ============================================================

def route_after_summary_classification(
    state: AgentState,
):

    if state.get("extraction_success"):
        return "end"

    print(
        "[GRAPH] Route -> repeat (unclear summary reply)"
    )

    return "repeat"


# ============================================================
# Advance
# ============================================================

async def advance_question(
    state: AgentState,
) -> AgentState:

    current_index = state[
        "current_question_index"
    ]

    next_index = current_index + 1

    next_question = (
        interview_manager.get_question(
            next_index
        )
    )

    # questions.json now has two system_message entries appended
    # after the real interview questions (the retry clarification
    # lines). Walking into one of those by raw index means we've
    # actually run out of real questions to ask — treat it exactly
    # like next_question being None.
    if (
        next_question is not None
        and next_question.get("type") == "system_message"
    ):

        print(
            "[GRAPH] Reached system_message entry"
            f" ({next_question['id']}) — treating as"
            " end of real questions"
        )

        next_question = None

    print(
        "[GRAPH] Advancing from:",
        (
            state["current_question"]["id"]
            if state.get("current_question")
            else None
        ),
    )

    print(
        "[GRAPH] Next question:",
        (
            next_question["id"]
            if next_question
            else None
        ),
    )

    # --------------------------------------------------------
    # No more questions — routed to build_summary next, which
    # will set current_question/response/mode appropriately.
    # --------------------------------------------------------

    if next_question is None:

        return {
            **state,
            "current_question": None,
            "current_question_index": next_index,
            "transcript": "",
        }

    # --------------------------------------------------------
    # Continue interview
    # --------------------------------------------------------

    return {
        **state,
        "current_question": next_question,
        "current_question_index": next_index,
        "response": next_question["question_ar"],
        "transcript": "",
        "extraction_success": False,
        "failure_reason": None,
    }


# ============================================================
# Repeat
# ============================================================

async def repeat_question(
    state: AgentState,
) -> AgentState:

    current_question = state.get(
        "current_question"
    )

    mode = state.get(
        "mode",
        "interview",
    )

    # --------------------------------------------------------
    # Unclear reply to the SUMMARY (not a real question) — just
    # replay it as-is. No clarifying prefix here; that's specific
    # to rejected answers on real interview questions.
    # --------------------------------------------------------

    if mode == "summary" or current_question is None:

        print(
            "[GRAPH] Repeating (as-is):",
            (
                current_question["id"]
                if current_question
                else None
            ),
        )

        return {
            **state,
            "current_question": current_question,
            "current_question_index": state[
                "current_question_index"
            ],
            "response": (
                current_question["question_ar"]
                if current_question
                else ""
            ),
            "transcript": "",
            "extraction_success": False,
        }

    # --------------------------------------------------------
    # Rejected answer to a real question — always rebuild from the
    # pristine question text (never from current_question, which
    # may already carry a clarifying prefix from a previous retry)
    # so consecutive failures don't stack prefixes on top of each
    # other.
    # --------------------------------------------------------

    base_id = _base_question_id(
        current_question["id"]
    )

    base_question = (
        _find_question_by_id(base_id)
        or current_question
    )

    failure_reason = (
        state.get("failure_reason")
        or "no_speech"
    )

    prefix = _retry_prefix_text(failure_reason)

    retry_text = build_retry_text(
        base_question["question_ar"],
        prefix,
    )

    retry_question = {
        **base_question,
        "id": f"{base_id}__retry_{failure_reason}",
        "question_ar": retry_text,
    }

    print(
        "[GRAPH] Repeating with clarification:",
        retry_question["id"],
    )

    return {
        **state,
        "current_question": retry_question,
        "current_question_index": state[
            "current_question_index"
        ],
        "response": retry_text,
        "transcript": "",
        "extraction_success": False,
    }


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)


builder.add_node(
    "initialize_interview",
    initialize_interview,
)

builder.add_node(
    "extract_answer",
    extract_answer,
)

builder.add_node(
    "advance_question",
    advance_question,
)

builder.add_node(
    "repeat_question",
    repeat_question,
)

builder.add_node(
    "build_summary",
    build_summary,
)

builder.add_node(
    "classify_summary",
    classify_summary,
)


# ============================================================
# IMPORTANT:
# START must be conditional
# ============================================================

builder.add_conditional_edges(
    START,
    route_from_start,
    {
        "initialize": "initialize_interview",
        "extract": "extract_answer",
        "classify_summary": "classify_summary",
    },
)


# Initialization ends after setting welcome
builder.add_edge(
    "initialize_interview",
    END,
)


# ============================================================
# Extraction routing
# ============================================================

builder.add_conditional_edges(
    "extract_answer",
    route_after_extraction,
    {
        "advance": "advance_question",
        "repeat": "repeat_question",
        "summarize": "build_summary",
    },
)


# ============================================================
# Advance routing
# ============================================================

builder.add_conditional_edges(
    "advance_question",
    route_after_advance,
    {
        "summarize": "build_summary",
        "end": END,
    },
)


builder.add_edge(
    "repeat_question",
    END,
)

builder.add_edge(
    "build_summary",
    END,
)


# ============================================================
# Summary classification routing
# ============================================================

builder.add_conditional_edges(
    "classify_summary",
    route_after_summary_classification,
    {
        "repeat": "repeat_question",
        "end": END,
    },
)


agent_graph = builder.compile()
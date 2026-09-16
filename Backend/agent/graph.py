from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import (
    InterviewManager,
    COMPLETION_ID,
    INTRO_ID,
    field_prompt_id,
)
from Backend.agent.extraction import (
    extract_all,
    extract_one,
    is_obviously_invalid,
)


QUESTIONS_PATH = "Backend/agent/questions.json"

interview_manager = InterviewManager(QUESTIONS_PATH)


# ============================================================
# Helpers
# ============================================================

def _next_prompt(state: AgentState, first_followup: bool) -> AgentState:
    """
    Decide what to say next: the next missing required field, or
    the completion line if nothing is missing.

    This is the single place that decides "are we done yet" — both
    the intro path and the follow-up path end here, so the rule
    lives in exactly one function.
    """

    candidate = state.get("candidate", {})

    missing = interview_manager.missing_required(candidate)

    print(
        "[GRAPH] Missing required fields:",
        [field["name"] for field in missing],
    )

    # --------------------------------------------------------
    # Nothing missing — finish.
    # --------------------------------------------------------

    if not missing:

        return {
            **state,
            "mode": "finished",
            "current_field": None,
            "current_prompt": {
                "id": COMPLETION_ID,
                "text": interview_manager.completion_text,
            },
            "interview_finished": True,
            "extraction_success": True,
            "failure_reason": None,
            "transcript": "",
            "first_followup": False,
        }

    # --------------------------------------------------------
    # Ask the next missing field.
    # --------------------------------------------------------

    field = missing[0]

    return {
        **state,
        "mode": "filling",
        "current_field": field["name"],
        "current_prompt": {
            "id": field_prompt_id(field["name"]),
            "text": field["question"],
        },
        "interview_finished": False,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
        "first_followup": first_followup,
    }


def _repeat_current(
    state: AgentState,
    failure_reason: str,
) -> AgentState:
    """
    Keep the same prompt and re-ask it. The prompt object is
    unchanged, so handler.py replays its cached audio — the
    clarification line is spoken separately, as its own cached
    clip, never merged into new text.
    """

    return {
        **state,
        "extraction_success": False,
        "failure_reason": failure_reason,
        "transcript": "",
        "first_followup": False,
    }


# ============================================================
# Nodes
# ============================================================

async def start_interview(state: AgentState) -> AgentState:
    """
    First invocation: speak the intro paragraph and wait for the
    candidate's long answer.
    """

    print("[GRAPH] Starting interview (one-paragraph mode)")

    return {
        **state,
        "candidate": {},
        "mode": "intro",
        "current_field": None,
        "current_prompt": {
            "id": INTRO_ID,
            "text": interview_manager.intro_text,
        },
        "interview_finished": False,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
        "first_followup": False,
    }


async def process_intro(state: AgentState) -> AgentState:
    """
    The candidate's opening paragraph: one LLM call extracts every
    field they mentioned, then we move straight to asking whatever
    is still missing.
    """

    transcript = state.get("transcript", "").strip()

    if is_obviously_invalid(transcript):

        print("[GRAPH] Intro answer unusable, repeating intro")

        return _repeat_current(state, "no_speech")

    updates = await extract_all(
        transcript,
        interview_manager.fields,
    )

    candidate = {**state.get("candidate", {}), **updates}

    print("[GRAPH] Candidate after intro:", candidate)

    # Nothing at all came out of a long answer — treat it as not
    # having answered rather than silently moving on, so the
    # candidate gets told why they're being asked again.
    if not updates:

        return _repeat_current(
            {**state, "candidate": candidate},
            "wrong_answer",
        )

    return _next_prompt(
        {**state, "candidate": candidate},
        first_followup=True,
    )


async def process_field(state: AgentState) -> AgentState:
    """
    An answer to one targeted follow-up question.
    """

    transcript = state.get("transcript", "").strip()

    field_name = state.get("current_field")

    field = interview_manager.get_field(field_name) if field_name else None

    if field is None:

        print("[GRAPH] No current field — re-deciding next prompt")

        return _next_prompt(state, first_followup=False)

    if is_obviously_invalid(transcript):

        print("[GRAPH] Answer unusable:", field_name)

        return _repeat_current(state, "no_speech")

    updates = await extract_one(transcript, field)

    if not updates:

        # Real speech reached us (is_obviously_invalid already
        # filtered out silence/filler above), so by elimination
        # this is "said something, just not an answer to this".
        print("[GRAPH] Answer rejected for:", field_name)

        return _repeat_current(state, "wrong_answer")

    candidate = {**state.get("candidate", {}), **updates}

    print("[GRAPH] Candidate after", field_name, ":", candidate)

    return _next_prompt(
        {**state, "candidate": candidate},
        first_followup=False,
    )


# ============================================================
# Routing
# ============================================================

def route_from_start(state: AgentState):

    transcript = state.get("transcript", "").strip()

    mode = state.get("mode", "intro")

    print("[GRAPH] START | mode:", mode, "| transcript:", transcript)

    # No transcript yet -> this is the opening invocation.
    if not transcript:
        return "start"

    if mode == "intro":
        return "intro"

    return "field"


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("start_interview", start_interview)
builder.add_node("process_intro", process_intro)
builder.add_node("process_field", process_field)

builder.add_conditional_edges(
    START,
    route_from_start,
    {
        "start": "start_interview",
        "intro": "process_intro",
        "field": "process_field",
    },
)

builder.add_edge("start_interview", END)
builder.add_edge("process_intro", END)
builder.add_edge("process_field", END)


agent_graph = builder.compile()
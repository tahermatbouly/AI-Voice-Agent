from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import InterviewManager, NONE_VALUE
from Backend.agent.extraction import (
    build_summary_segments,
    classify_summary_reply,
    is_obviously_invalid,
    process_turn,
)


QUESTIONS_PATH = "Backend/agent/questions.json"

interview_manager = InterviewManager(QUESTIONS_PATH)


# ============================================================
# Segment resolution
# ============================================================
# Turns a question's raw segment list (which may contain "value"
# segments referencing a field) into what handler.py actually
# speaks — plain {"kind": "static"/"value", ...} dicts with every
# value already substituted from the candidate dict. Keeping this
# resolution in the graph means handler.py never touches
# state["candidate"] directly.

def _resolve_segments(
    segments: list[dict],
    candidate: dict,
) -> list[dict]:

    resolved = []

    for seg in segments:

        if seg["type"] == "static":

            resolved.append(
                {"kind": "static", "id": seg["id"], "text": seg["text"]}
            )

        elif seg["type"] == "value":

            value = candidate.get(seg["field"])

            if value is None or value == NONE_VALUE:
                continue

            resolved.append(
                {"kind": "value", "text": _format(value)}
            )

        elif seg["type"] == "value_text":

            # Already-formatted text (used by build_summary_segments,
            # which resolves the value itself before this point).
            resolved.append({"kind": "value", "text": seg["text"]})

    return resolved


def _format(value) -> str:

    if isinstance(value, list):
        return "، ".join(str(item) for item in value)

    return str(value)


def _plain_text(segments: list[dict], candidate: dict) -> str:
    """
    The full sentence actually spoken for these segments, used as
    the CURRENT QUESTION context in the extraction prompt. Built by
    resolving and joining every segment (not just assuming the last
    one is static) so this stays correct regardless of where a
    "value" segment falls in the sentence.
    """

    resolved = _resolve_segments(segments, candidate)

    return " ".join(seg["text"] for seg in resolved)


def _question_prompt(
    question: dict,
    candidate: dict,
    repeat: bool = False,
) -> list[dict]:

    segments = (
        question.get("repeat_segments") or question["segments"]
        if repeat
        else question["segments"]
    )

    return _resolve_segments(segments, candidate)


# ============================================================
# Start
# ============================================================

async def start_interview(state: AgentState) -> AgentState:

    print("[GRAPH] Starting interview")

    first_question = interview_manager.get_question(0)

    return {
        **state,
        "candidate": {},
        "current_question_index": 0,
        "current_question": first_question,
        "mode": "interview",
        "current_prompt": _question_prompt(first_question, {}),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "interview_finished": False,
        "ended_early": False,
        "transcript": "",
    }


# ============================================================
# Advance / summary helpers
# ============================================================

def _advance_or_summarize(state: AgentState) -> AgentState:

    candidate = state["candidate"]

    next_index = state["current_question_index"] + 1

    next_question = interview_manager.get_question(next_index)

    if next_question is None:

        print("[GRAPH] Out of questions -> summary")

        return _build_summary(state)

    print("[GRAPH] Advancing to:", next_question["id"])

    return {
        **state,
        "current_question_index": next_index,
        "current_question": next_question,
        "mode": "interview",
        "current_prompt": _question_prompt(next_question, candidate),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
    }


def _build_summary(state: AgentState) -> AgentState:

    segments = build_summary_segments(
        state["candidate"], interview_manager
    )

    return {
        **state,
        "current_question": None,
        "mode": "summary",
        "current_prompt": _resolve_segments(segments, state["candidate"]),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
    }


def _retry_or_skip(state: AgentState, failure_reason: str) -> AgentState:
    """
    Repeats the current question, unless it has already failed
    max_retries_per_question times in a row — then it moves on
    rather than looping forever. Required fields still get skipped
    past this limit (better an incomplete interview than a stuck
    call); skip_optional_after_max_retries only changes whether an
    optional field's absence is worth distinguishing in logs, since
    either way we must advance.
    """

    question = state["current_question"]
    candidate = state["candidate"]

    retry_count = state["retry_count"] + 1

    if retry_count >= interview_manager.max_retries_per_question:

        print(
            "[GRAPH] Max retries reached for",
            question["id"] if question else None,
            "- skipping",
        )

        return _advance_or_summarize({**state, "retry_count": 0})

    return {
        **state,
        "current_prompt": _question_prompt(
            question, candidate, repeat=True
        ),
        "intent_prompt": [],
        "retry_count": retry_count,
        "extraction_success": False,
        "failure_reason": failure_reason,
        "transcript": "",
    }


# ============================================================
# Process an answer to a normal (or correcting) question
# ============================================================

async def process_answer(state: AgentState) -> AgentState:

    question = state["current_question"]
    transcript = state["transcript"].strip()
    candidate = dict(state["candidate"])

    # ----------------------------------------------------
    # Obvious silence/filler — skip the LLM call entirely.
    # ----------------------------------------------------

    if is_obviously_invalid(transcript):

        print("[GRAPH] Obviously invalid transcript")

        return _retry_or_skip(state, "no_speech")

    target_field_names = interview_manager.target_fields(question)

    fields = [
        interview_manager.get_field(name) for name in target_field_names
    ]
    fields = [f for f in fields if f is not None]

    result = await process_turn(
        question_text=_plain_text(question["segments"], candidate),
        fields=fields,
        intents=interview_manager.intents,
        transcript=transcript,
    )

    print("[GRAPH] Turn result:", result)

    # ----------------------------------------------------
    # Intent matched — not an answer at all.
    # ----------------------------------------------------

    if result["kind"] == "intent":

        return await _handle_intent(state, result["intent_id"])

    # ----------------------------------------------------
    # No usable answer, no intent match.
    # ----------------------------------------------------

    if result["kind"] != "answer" or not result["updates"]:

        return _retry_or_skip(state, "wrong_answer")

    # ----------------------------------------------------
    # Real answer.
    # ----------------------------------------------------

    candidate.update(result["updates"])

    new_state = {**state, "candidate": candidate}

    if state["mode"] == "correcting":

        print("[GRAPH] Correction applied, rebuilding summary")

        return _build_summary(new_state)

    return _advance_or_summarize(new_state)


# ============================================================
# Intents
# ============================================================

async def _handle_intent(state: AgentState, intent_id: str) -> AgentState:

    intent = interview_manager.get_intent(intent_id)

    if intent is None:
        # Unknown id somehow — fall back to a normal retry rather
        # than crash.
        return _retry_or_skip(state, "wrong_answer")

    intent_prompt = [
        {"kind": "static", "id": intent["id"], "text": intent["text"]}
    ]

    action = intent.get("then", "repeat_question")

    question = state["current_question"]
    candidate = state["candidate"]

    print("[GRAPH] Intent:", intent_id, "| then:", action)

    # --------------------------------------------------------
    # wait: say the line, keep the exact same question active,
    # don't touch retry_count (asking for a moment isn't a failed
    # answer) and don't repeat the question — silence afterward is
    # just them thinking.
    # --------------------------------------------------------

    if action == "wait":

        return {
            **state,
            "current_prompt": [],
            "intent_prompt": intent_prompt,
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    # --------------------------------------------------------
    # repeat_question: say the line, then repeat the question —
    # doesn't count against the retry limit, this isn't a failed
    # extraction attempt.
    # --------------------------------------------------------

    if action == "repeat_question":

        return {
            **state,
            "current_prompt": _question_prompt(
                question, candidate, repeat=True
            ),
            "intent_prompt": intent_prompt,
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    # --------------------------------------------------------
    # skip_question: say the line, move on without filling this
    # question's fields.
    # --------------------------------------------------------

    if action == "skip_question":

        advanced = _advance_or_summarize({**state, "retry_count": 0})

        return {**advanced, "intent_prompt": intent_prompt}

    # --------------------------------------------------------
    # end_call: say the line (it's already a full goodbye), stop
    # here. handler.py sees ended_early and saves without playing
    # the normal completion clip.
    # --------------------------------------------------------

    if action == "end_call":

        return {
            **state,
            "current_prompt": [],
            "intent_prompt": intent_prompt,
            "mode": "finished",
            "extraction_success": True,
            "failure_reason": None,
            "interview_finished": True,
            "ended_early": True,
            "transcript": "",
        }

    return _retry_or_skip(state, "wrong_answer")


# ============================================================
# Summary
# ============================================================

async def process_summary_reply(state: AgentState) -> AgentState:

    transcript = state["transcript"].strip()

    if is_obviously_invalid(transcript):

        print("[GRAPH] Summary reply obviously invalid, replaying")

        return {
            **state,
            "current_prompt": _resolve_segments(
                build_summary_segments(
                    state["candidate"], interview_manager
                ),
                state["candidate"],
            ),
            "intent_prompt": [],
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    decision = await classify_summary_reply(
        transcript, interview_manager.fields
    )

    print("[GRAPH] Summary reply:", decision)

    if decision["action"] == "confirm":

        candidate = state["candidate"]

        return {
            **state,
            "mode": "finished",
            "current_question": None,
            "current_prompt": _resolve_segments(
                interview_manager.completion_segments, candidate
            ),
            "intent_prompt": [],
            "extraction_success": True,
            "failure_reason": None,
            "interview_finished": True,
            "ended_early": False,
            "transcript": "",
        }

    if decision["action"] == "correct":

        target_question = interview_manager.question_for_field(
            decision["field"]
        )

        if target_question is not None:

            print("[GRAPH] Correcting:", target_question["id"])

            return {
                **state,
                "mode": "correcting",
                "current_question": target_question,
                "current_prompt": _question_prompt(
                    target_question, state["candidate"], repeat=True
                ),
                "intent_prompt": [],
                "retry_count": 0,
                "extraction_success": True,
                "failure_reason": None,
                "transcript": "",
            }

    print("[GRAPH] Summary reply unclear, replaying summary")

    return {
        **state,
        "current_prompt": _resolve_segments(
            build_summary_segments(
                state["candidate"], interview_manager
            ),
            state["candidate"],
        ),
        "intent_prompt": [],
        "extraction_success": False,
        "failure_reason": None,
        "transcript": "",
    }


# ============================================================
# Routing
# ============================================================

def route_from_start(state: AgentState):
    """
    current_question_index starts at -1 in handler.py's initial
    state dict and is only ever set to >= 0 by start_interview.
    That — NOT "is the transcript empty" — is what actually
    distinguishes "this is the very first call" from "this is a
    normal turn where STT happened to come back empty".

    An empty transcript mid-interview must NOT re-trigger
    start_interview: that would silently wipe state["candidate"]
    and restart the whole call from question 0. process_answer's
    and process_summary_reply's own is_obviously_invalid() checks
    already treat "" correctly as a no_speech retry — there's
    nothing left for this router to special-case.
    """

    if state.get("current_question_index", -1) < 0:
        return "start"

    mode = state.get("mode", "interview")

    if mode == "summary":
        return "summary"

    return "answer"


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("start_interview", start_interview)
builder.add_node("process_answer", process_answer)
builder.add_node("process_summary_reply", process_summary_reply)

builder.add_conditional_edges(
    START,
    route_from_start,
    {
        "start": "start_interview",
        "answer": "process_answer",
        "summary": "process_summary_reply",
    },
)

builder.add_edge("start_interview", END)
builder.add_edge("process_answer", END)
builder.add_edge("process_summary_reply", END)

agent_graph = builder.compile()
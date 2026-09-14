from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import InterviewManager
from Backend.agent.extraction import (
    extract_candidate_info,
    build_summary_text,
    classify_summary_reply,
)


QUESTIONS_PATH = "Backend/agent/questions.json"
SUMMARY_QUESTION_ID = "summary"

interview_manager = InterviewManager(QUESTIONS_PATH)


# ---------------------------------------------------------
# Initialize
# ---------------------------------------------------------

async def initialize_interview(state: AgentState):
    welcome = interview_manager.get_question(0)

    return {
        **state,
        "current_question_index": 0,
        "current_question": welcome,
        "response": welcome["question_ar"],
        "transcript": "",
        "extraction_success": True,
        "failure_reason": None,
        "interview_finished": False,
        "mode": "interview",
    }


# ---------------------------------------------------------
# Extract candidate answer
# ---------------------------------------------------------

async def extract_answer(state: AgentState):
    return await extract_candidate_info(state)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _find_question_by_field(field: str):
    for index, question in enumerate(interview_manager.questions):
        if question.get("target_field") == field:
            return index, question

        if field in question.get("target_fields", []):
            return index, question

    return None, None


# ---------------------------------------------------------
# Build summary
# ---------------------------------------------------------

async def build_summary(state: AgentState):
    candidate = state.get("candidate", {})

    summary_text = build_summary_text(candidate)

    summary_question = {
        "id": SUMMARY_QUESTION_ID,
        "question_ar": summary_text,
    }

    return {
        **state,
        "current_question": summary_question,
        "response": summary_text,
        "transcript": "",
        "extraction_success": True,
        "failure_reason": None,
        "interview_finished": False,
        "mode": "summary",
    }


# ---------------------------------------------------------
# Classify summary response
# ---------------------------------------------------------

async def classify_summary(state: AgentState):
    transcript = state.get("transcript", "").strip()

    decision = await classify_summary_reply(transcript)

    # Candidate confirmed the information
    if decision.get("action") == "confirm":
        return {
            **state,
            "transcript": "",
            "extraction_success": True,
            "failure_reason": None,
            "interview_finished": True,
        }

    # Candidate wants to correct something
    if decision.get("action") == "correct":
        field = decision.get("field")

        question_index, question = _find_question_by_field(field)

        if question is not None:
            return {
                **state,
                "current_question_index": question_index,
                "current_question": question,
                "response": question["question_ar"],
                "transcript": "",
                "extraction_success": True,
                "failure_reason": None,
                "interview_finished": False,
                "mode": "correcting",
            }

    # Could not understand the summary response
    return {
        **state,
        "transcript": "",
        "extraction_success": False,
        "failure_reason": "wrong_answer",
        "interview_finished": False,
        "mode": "summary",
    }


# ---------------------------------------------------------
# Move to next question
# ---------------------------------------------------------

async def advance_question(state: AgentState):
    current_index = state["current_question_index"]
    next_index = current_index + 1

    # No more questions
    if next_index >= len(interview_manager.questions):
        return {
            **state,
            "current_question_index": next_index,
            "current_question": None,
            "response": "",
            "transcript": "",
            "extraction_success": True,
            "failure_reason": None,
        }

    next_question = interview_manager.questions[next_index]

    # System messages are not interview questions.
    # Reaching one means all real questions are finished.
    if next_question.get("type") == "system_message":
        return {
            **state,
            "current_question_index": next_index,
            "current_question": None,
            "response": "",
            "transcript": "",
            "extraction_success": True,
            "failure_reason": None,
        }

    return {
        **state,
        "current_question_index": next_index,
        "current_question": next_question,
        "response": next_question["question_ar"],
        "transcript": "",
        "extraction_success": True,
        "failure_reason": None,
        "mode": "interview",
    }


# ---------------------------------------------------------
# Repeat current question
# ---------------------------------------------------------

async def repeat_question(state: AgentState):
    current_question = state.get("current_question")

    if current_question is None:
        return {
            **state,
            "transcript": "",
            "extraction_success": False,
        }

    return {
        **state,
        # Keep the original question unchanged.
        # The retry phrase is handled separately by handler.py.
        "current_question": current_question,
        "response": current_question["question_ar"],
        "transcript": "",
        "extraction_success": False,
        "failure_reason": state.get("failure_reason") or "wrong_answer",
    }


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

def route_from_start(state: AgentState):
    # First call: initialize the interview
    if state.get("current_question") is None:
        return "initialize"

    # Summary response
    if state.get("mode") == "summary":
        return "classify_summary"

    # Normal interview/correction answer
    if state.get("transcript"):
        return "extract"

    return "initialize"


def route_after_extraction(state: AgentState):
    if not state.get("extraction_success"):
        return "repeat"

    # Correction was successfully extracted.
    if state.get("mode") == "correcting":
        return "summary"

    return "advance"


def route_after_advance(state: AgentState):
    if state.get("current_question") is None:
        return "summary"

    return END


def route_after_summary(state: AgentState):
    if state.get("extraction_success"):
        return END

    return "repeat"


# ---------------------------------------------------------
# Build graph
# ---------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("initialize", initialize_interview)
graph.add_node("extract", extract_answer)
graph.add_node("advance", advance_question)
graph.add_node("repeat", repeat_question)
graph.add_node("summary", build_summary)
graph.add_node("classify_summary", classify_summary)

graph.add_conditional_edges(
    START,
    route_from_start,
    {
        "initialize": "initialize",
        "extract": "extract",
        "classify_summary": "classify_summary",
    },
)

graph.add_edge("initialize", END)

graph.add_conditional_edges(
    "extract",
    route_after_extraction,
    {
        "repeat": "repeat",
        "advance": "advance",
        "summary": "summary",
    },
)

graph.add_edge("repeat", END)

graph.add_conditional_edges(
    "advance",
    route_after_advance,
    {
        "summary": "summary",
        END: END,
    },
)

graph.add_edge("summary", END)

graph.add_conditional_edges(
    "classify_summary",
    route_after_summary,
    {
        "repeat": "repeat",
        END: END,
    },
)


agent_graph = graph.compile()
from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import InterviewManager
from Backend.agent.extraction import extract_candidate_info


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

    print(
        "[GRAPH] Extraction success:",
        success,
    )

    if success:

        print(
            "[GRAPH] Route -> advance"
        )

        return "advance"

    print(
        "[GRAPH] Route -> repeat"
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
    # Interview finished
    # --------------------------------------------------------

    if next_question is None:

        return {
            **state,
            "current_question": None,
            "current_question_index": next_index,
            "response": "",
            "transcript": "",
            "interview_finished": True,
            "extraction_success": True,
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
    }


# ============================================================
# Repeat
# ============================================================

async def repeat_question(
    state: AgentState,
) -> AgentState:

    current_question = state[
        "current_question"
    ]

    print(
        "[GRAPH] Repeating:",
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
    },
)


builder.add_edge(
    "advance_question",
    END,
)

builder.add_edge(
    "repeat_question",
    END,
)


agent_graph = builder.compile()
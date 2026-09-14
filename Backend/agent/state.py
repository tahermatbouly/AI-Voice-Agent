from typing import Literal, TypedDict


class CandidateInfo(TypedDict, total=False):
    candidate_name: str
    target_domain: str
    years_of_experience: str
    education_level: str
    key_skills: list[str]
    tools_technologies: list[str]
    english_proficiency: str


class AgentState(TypedDict):
    transcript: str

    candidate: CandidateInfo

    questions: list[dict]

    current_question_index: int

    current_question: dict | None

    response: str

    interview_finished: bool

    extraction_success: bool

    # Why the last answer was rejected: "no_speech" (no usable
    # transcript) or "wrong_answer" (real speech, wrong question).
    # None when there's nothing to report (success, or before the
    # first answer). Without this declared here, LangGraph has no
    # channel for it — any node returning it gets the value silently
    # dropped on the merge, which is why repeat_question() was always
    # falling back to its "no_speech" default.
    failure_reason: Literal["no_speech", "wrong_answer"] | None

    # Which phase of the conversation we're in — drives routing in
    # graph.py. "interview": walking through questions.json normally.
    # "summary": the candidate is replying to the spoken summary.
    # "correcting": they asked to change one field and are now
    # re-answering that specific question.
    mode: Literal["interview", "summary", "correcting"]
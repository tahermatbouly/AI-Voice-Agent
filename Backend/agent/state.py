from typing import TypedDict


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
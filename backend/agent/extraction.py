"""
Structured candidate information extraction.

Canonical candidate schema:

    candidate_name
    target_domain
    years_of_experience
    education_level
    key_skills
    tools_technologies
    english_proficiency
    notes

The same schema is used by:
    - CallExtraction
    - ExtractionTools
    - QuestionManager
    - final candidate record
"""

from __future__ import annotations

from typing import Optional

from livekit.agents import RunContext, function_tool


# ============================================================
# CANONICAL FIELDS
# ============================================================

EXTRACTION_FIELDS = (
    "candidate_name",
    "target_domain",
    "years_of_experience",
    "education_level",
    "key_skills",
    "tools_technologies",
    "english_proficiency",
    "notes",
)


# ============================================================
# CALL EXTRACTION
# ============================================================

class CallExtraction:
    """
    Holds structured information for one call.

    Create one instance for every call.
    """

    def __init__(self) -> None:

        self.record: dict[str, str] = {
            field: ""
            for field in EXTRACTION_FIELDS
        }

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        candidate_name: Optional[str] = None,
        target_domain: Optional[str] = None,
        years_of_experience: Optional[str] = None,
        education_level: Optional[str] = None,
        key_skills: Optional[str] = None,
        tools_technologies: Optional[str] = None,
        english_proficiency: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:

        updates = {
            "candidate_name": candidate_name,
            "target_domain": target_domain,
            "years_of_experience": years_of_experience,
            "education_level": education_level,
            "key_skills": key_skills,
            "tools_technologies": tools_technologies,
            "english_proficiency": english_proficiency,
            "notes": notes,
        }

        for field, value in updates.items():

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            self.record[field] = value

    # --------------------------------------------------------
    # GET RECORD
    # --------------------------------------------------------

    def get_record(self) -> dict[str, str]:

        return self.record.copy()

    # --------------------------------------------------------
    # GET MISSING
    # --------------------------------------------------------

    def get_missing_fields(self) -> list[str]:

        return [
            field
            for field in EXTRACTION_FIELDS
            if not self.record[field]
        ]

    # --------------------------------------------------------
    # CHECK INFORMATION
    # --------------------------------------------------------

    def has_information(self) -> bool:

        return any(
            bool(value.strip())
            for value in self.record.values()
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:

        for field in EXTRACTION_FIELDS:
            self.record[field] = ""


# ============================================================
# EXTRACTION TOOLS
# ============================================================

class ExtractionTools:
    """
    Tools exposed to the conversational LLM.

    The LLM uses update_candidate_info whenever the caller
    explicitly provides candidate information.
    """

    def __init__(
        self,
        extraction: CallExtraction,
    ) -> None:

        self.extraction = extraction

    @function_tool
    async def update_candidate_info(
        self,
        context: RunContext,
        candidate_name: str | None = None,
        target_domain: str | None = None,
        years_of_experience: str | None = None,
        education_level: str | None = None,
        key_skills: str | None = None,
        tools_technologies: str | None = None,
        english_proficiency: str | None = None,
        notes: str | None = None,
    ) -> str:
        """
        Save information explicitly stated by the caller.

        Rules:

        - Only save information explicitly provided.
        - Never guess or infer missing information.
        - Multiple fields may be updated from one message.
        - Only provide fields relevant to the latest message.
        - Existing information may be replaced if the caller
          clearly corrects it.
        """

        self.extraction.update(
            candidate_name=candidate_name,
            target_domain=target_domain,
            years_of_experience=years_of_experience,
            education_level=education_level,
            key_skills=key_skills,
            tools_technologies=tools_technologies,
            english_proficiency=english_proficiency,
            notes=notes,
        )

        return "Candidate information updated."
from typing import Any, Literal, TypedDict


class AgentState(TypedDict):
    """
    Every key here must be declared — LangGraph builds its state
    channels from this TypedDict, and any key a node returns that
    isn't listed gets silently dropped on the merge.
    """

    transcript: str

    # Collected answers. Keys are the "name" values from
    # questions.json, so this is a plain dict rather than a fixed
    # TypedDict — the field set is config-driven now.
    candidate: dict[str, Any]

    # "intro"    — waiting for / processing the opening paragraph
    # "filling"  — asking one missing field at a time
    # "finished" — everything required collected
    mode: Literal["intro", "filling", "finished"]

    # Name of the field currently being asked about. None during
    # the intro turn.
    current_field: str | None

    # The clip to speak next: {"id": ..., "text": ...}. handler.py
    # speaks this by id, so it's always a cache hit.
    current_prompt: dict | None

    # True when the last turn produced at least one new value.
    extraction_success: bool

    # Why the last turn produced nothing: "no_speech" (nothing
    # usable was transcribed) or "wrong_answer" (real speech that
    # didn't answer what was asked). None on success.
    failure_reason: Literal["no_speech", "wrong_answer"] | None

    # Set once, at the very end. handler.py saves the JSON and
    # speaks the completion line when it sees this.
    interview_finished: bool

    # True only on the follow-up prompt immediately after the intro
    # paragraph, so handler.py knows to play the one-off
    # "let me ask about what's missing" line before it.
    first_followup: bool
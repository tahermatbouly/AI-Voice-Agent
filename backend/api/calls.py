"""
REST endpoints for the dashboard frontend.

Provides:
- A lightweight list of past calls
- Full details and transcript for one call
- Raw audio for playback/download

The API layer delegates all database operations to app.db.
It does not contain voice-agent, STT, TTS, LLM, extraction,
LiveKit, or memory logic.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app import db


router = APIRouter()


@router.get("/calls")
def get_calls():
    """
    List all completed calls, most recent first.

    Used by the dashboard's call-history page.

    The database layer deliberately returns only lightweight
    summary information here -- no transcript or raw audio.
    """
    return db.list_calls()


@router.get("/calls/{call_id}")
def get_call_detail(call_id: int):
    """
    Get the complete information for one call.

    Returns:
    - Extracted HR candidate information
    - Full conversation transcript
    - Whether a recording is available
    """
    call = db.get_call(call_id)

    if call is None:
        raise HTTPException(
            status_code=404,
            detail="Call not found",
        )

    return call


@router.get("/calls/{call_id}/audio")
def get_call_audio(call_id: int):
    """
    Return the raw audio recording for one call.

    Audio is intentionally served through a separate binary endpoint
    instead of being embedded in the JSON returned by get_call().
    This allows the frontend to use the endpoint directly as the
    source of an HTML <audio> element.
    """
    audio = db.get_call_audio(call_id)

    if audio is None:
        raise HTTPException(
            status_code=404,
            detail="Call not found or recording not available",
        )

    audio_bytes, audio_format = audio

    # Map the formats currently supported by the database layer
    # to proper HTTP Content-Type values.
    content_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
        "m4a": "audio/mp4",
    }

    media_type = content_types.get(
        audio_format.lower(),
        "application/octet-stream",
    )

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'inline; filename="call_{call_id}.{audio_format}"'
            )
        },
    )

import logging

from fastapi import WebSocket, WebSocketDisconnect

from Backend.audio.audio_utils import (
    TARGET_SAMPLE_RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_audio,
)
from Backend.audio.buffer import AudioBuffer
from Backend.audio.vad import VoiceActivityDetector
from Backend.stt_plugin import CohereArabicSTT

from Backend.agent.graph import agent_graph
from Backend.agent.interview_manager import InterviewManager
from Backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"


async def websocket_endpoint(
    websocket: WebSocket,
):

    await websocket.accept()

    logger.info("[WS] Client connected")

    # =====================================================
    # SESSION COMPONENTS
    # =====================================================

    audio_buffer = AudioBuffer()

    vad = VoiceActivityDetector()

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    interview_manager = InterviewManager(
        QUESTIONS_PATH
    )

    # =====================================================
    # INITIAL SESSION STATE
    # =====================================================

    state = {
        "transcript": "",
        "candidate": {},
        "questions": interview_manager.questions,
        "current_question_index": -1,
        "current_question": None,
        "response": "",
        "interview_finished": False,
        "extraction_success": False,
    }

    speech_active = False

    # =====================================================
    # TTS HELPER
    # =====================================================

    async def speak(text: str):

        if not text:
            return

        logger.info(
            "[TTS] Speaking: %s",
            text,
        )

        try:

            audio, sample_rate = await tts.synthesize(
                text
            )

            logger.info(
                "[TTS] Generated %d bytes at %d Hz",
                len(audio),
                sample_rate,
            )

            # Tell the client that TTS audio is coming
            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            # Send raw PCM audio
            await websocket.send_bytes(audio)

            # Tell the client that TTS audio is finished
            await websocket.send_json({
                "type": "tts_end",
            })

            logger.info(
                "[TTS] Audio sent to client"
            )

        except WebSocketDisconnect:
            raise

        except Exception as e:

            logger.exception(
                "[TTS] Synthesis failed"
            )

            try:
                await websocket.send_json({
                    "type": "error",
                    "stage": "tts",
                    "message": str(e),
                })
            except WebSocketDisconnect:
                raise

    try:

        # =================================================
        # START INTERVIEW
        # =================================================

        logger.info(
            "[INTERVIEW] Starting interview..."
        )

        # -------------------------------------------------
        # WELCOME
        # -------------------------------------------------

        welcome_question = interview_manager.get_question(0)

        if welcome_question is None:

            logger.warning(
                "[INTERVIEW] No welcome question found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state["current_question_index"] = 0
        state["current_question"] = welcome_question
        state["response"] = welcome_question["question_ar"]

        logger.info(
            "[INTERVIEW] Welcome: %s",
            welcome_question["id"],
        )

        await websocket.send_json({
            "type": "question",
            "id": welcome_question["id"],
            "text": welcome_question["question_ar"],
            "is_welcome": True,
        })

        # Speak welcome
        await speak(
            welcome_question["question_ar"]
        )

        # -------------------------------------------------
        # FIRST REAL QUESTION
        # -------------------------------------------------

        first_question_index = 1

        first_question = interview_manager.get_question(
            first_question_index
        )

        if first_question is None:

            logger.warning(
                "[INTERVIEW] No interview questions found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state["current_question_index"] = first_question_index
        state["current_question"] = first_question
        state["response"] = first_question["question_ar"]
        state["transcript"] = ""
        state["extraction_success"] = False

        logger.info(
            "[INTERVIEW] First question: %s",
            first_question["id"],
        )

        await websocket.send_json({
            "type": "question",
            "id": first_question["id"],
            "text": first_question["question_ar"],
            "is_welcome": False,
            "repeat": False,
        })

        # Speak first question
        await speak(
            first_question["question_ar"]
        )

        # =================================================
        # AUDIO LOOP
        # =================================================

        while True:

            message = await websocket.receive()

            # =================================================
            # TEXT MESSAGE
            # =================================================

            if message.get("text") is not None:

                text = message["text"]

                logger.info(
                    "[WS] Received text: %s",
                    text,
                )

                await websocket.send_json({
                    "type": "message_received",
                    "message": text,
                })

                continue

            # =================================================
            # AUDIO MESSAGE
            # =================================================

            audio_data = message.get("bytes")

            if audio_data is None:
                continue

            if not audio_data:
                continue

            # =================================================
            # NORMALIZE AUDIO
            # =================================================

            audio = pcm16_to_float32(
                audio_data
            )

            audio = resample_audio(
                audio,
                source_rate=INPUT_SAMPLE_RATE,
                target_rate=TARGET_SAMPLE_RATE,
            )

            normalized_audio = float32_to_pcm16(
                audio
            )

            # =================================================
            # VAD
            # =================================================

            vad_events = vad.process(
                normalized_audio
            )

            for event in vad_events:

                # =============================================
                # SPEECH START
                # =============================================

                if event == "speech_start":

                    logger.info(
                        "[VAD] Speech started"
                    )

                    speech_active = True

                    audio_buffer.clear()

                    await websocket.send_json({
                        "type": "speech_started"
                    })

                # =============================================
                # SPEECH END
                # =============================================

                elif event == "speech_end":

                    logger.info(
                        "[VAD] Speech ended | Audio: %d bytes",
                        audio_buffer.size(),
                    )

                    speech_active = False

                    utterance = audio_buffer.get_audio()

                    logger.info(
                        "[AUDIO] Complete utterance | %d bytes",
                        len(utterance),
                    )

                    await websocket.send_json({
                        "type": "speech_ended",
                        "bytes": len(utterance),
                        "sample_rate": TARGET_SAMPLE_RATE,
                    })

                    # =========================================
                    # STT
                    # =========================================

                    if utterance:

                        logger.info(
                            "[STT] Transcribing utterance..."
                        )

                        try:

                            transcript = await stt.transcribe(
                                audio_bytes=utterance,
                                sample_rate=TARGET_SAMPLE_RATE,
                            )

                            transcript = transcript.strip()

                            logger.info(
                                "[STT] Transcript: %s",
                                transcript or "<EMPTY>",
                            )

                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript,
                            })

                            # =================================
                            # LANGGRAPH
                            # =================================

                            if transcript:

                                logger.info(
                                    "[AGENT] Processing candidate answer..."
                                )

                                try:

                                    state["transcript"] = transcript
                                    state["extraction_success"] = False

                                    logger.info(
                                        "[AGENT] Current question: %s",
                                        state["current_question"]["id"],
                                    )

                                    result = await agent_graph.ainvoke(
                                        state
                                    )

                                    state = result

                                    logger.info(
                                        "[AGENT] Candidate: %s",
                                        state["candidate"],
                                    )

                                    logger.info(
                                        "[AGENT] Extraction success: %s",
                                        state["extraction_success"],
                                    )

                                    # =================================
                                    # INVALID ANSWER
                                    # =================================

                                    if not state["extraction_success"]:

                                        current_question = state[
                                            "current_question"
                                        ]

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Answer not understood. "
                                            "Repeating: %s",
                                            current_question["id"],
                                        )

                                        await websocket.send_json({
                                            "type": "answer_not_understood",
                                            "question_id": (
                                                current_question["id"]
                                            ),
                                        })

                                        await websocket.send_json({
                                            "type": "question",
                                            "id": current_question["id"],
                                            "text": (
                                                current_question[
                                                    "question_ar"
                                                ]
                                            ),
                                            "is_welcome": False,
                                            "repeat": True,
                                        })

                                        await speak(
                                            current_question[
                                                "question_ar"
                                            ]
                                        )

                                        continue

                                    # =================================
                                    # INTERVIEW FINISHED
                                    # =================================

                                    if state["interview_finished"]:

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Interview completed"
                                        )

                                        await websocket.send_json({
                                            "type": "interview_complete",
                                            "candidate": (
                                                state["candidate"]
                                            ),
                                        })

                                        return

                                    # =================================
                                    # NEXT QUESTION
                                    # =================================

                                    next_question = state[
                                        "current_question"
                                    ]

                                    if next_question:

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Next question: %s",
                                            next_question["id"],
                                        )

                                        await websocket.send_json({
                                            "type": "question",
                                            "id": next_question["id"],
                                            "text": (
                                                next_question[
                                                    "question_ar"
                                                ]
                                            ),
                                            "is_welcome": False,
                                            "repeat": False,
                                        })

                                        await speak(
                                            next_question[
                                                "question_ar"
                                            ]
                                        )

                                except WebSocketDisconnect:
                                    raise

                                except Exception as e:

                                    logger.exception(
                                        "[AGENT] Processing failed"
                                    )

                                    await websocket.send_json({
                                        "type": "error",
                                        "stage": "agent",
                                        "message": str(e),
                                    })

                        except WebSocketDisconnect:
                            raise

                        except Exception as e:

                            logger.exception(
                                "[STT] Transcription failed"
                            )

                            await websocket.send_json({
                                "type": "error",
                                "stage": "stt",
                                "message": str(e),
                            })

                    # -----------------------------------------
                    # CLEAN UP UTTERANCE
                    # -----------------------------------------

                    audio_buffer.clear()

                    vad.reset()

            # =================================================
            # STORE SPEECH AUDIO
            # =================================================

            if speech_active:

                audio_buffer.add(
                    normalized_audio
                )

    except WebSocketDisconnect:

        logger.info(
            "[WS] Client disconnected"
        )

    except RuntimeError as e:

        if "disconnect message" in str(e):

            logger.info(
                "[WS] Client disconnected"
            )

        else:

            raise

    finally:

        logger.info(
            "[WS] Connection closed"
        )

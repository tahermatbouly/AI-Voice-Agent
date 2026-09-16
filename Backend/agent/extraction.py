from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from Backend import config


# ============================================================
# LLM
# ============================================================
# temperature=0: this is extraction/classification, not writing.
# Any randomness here shows up as flaky tool-call behaviour on
# borderline answers.

llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ============================================================
# Known STT artifacts / obviously invalid answers
# ============================================================

INVALID_TRANSCRIPTS = {
    "",
    "لا",
    "لأ",
    "مش عارف",
    "مش عارفة",
    "معرفش",
    "ماعرفش",
    "ونكمل",
    "منور",
    "تمام",
    "نعم",
    "اه",
    "آه",
    "أيوه",
    "@@فراغ",
    "فراغ",
}


def is_obviously_invalid(transcript: str) -> bool:

    normalized = transcript.strip().lower()

    return normalized in INVALID_TRANSCRIPTS


# ============================================================
# Dynamic extraction tool
# ============================================================

def _build_tool(fields: list[dict]) -> StructuredTool:
    """
    Build a tool whose arguments are exactly the given fields.

    Used two ways:
      - every field at once, for the opening paragraph
      - a single field, for a targeted follow-up question

    Every argument is optional: the model fills in what it actually
    heard and leaves the rest null, which is what makes one call
    over a long paragraph possible.
    """

    model_fields = {}

    for field in fields:

        annotation = (
            list[str] | None
            if field.get("multi")
            else str | None
        )

        model_fields[field["name"]] = (
            annotation,
            Field(
                default=None,
                description=field["description"],
            ),
        )

    ExtractionModel = create_model(
        "CandidateInfo",
        **model_fields,
    )

    def save_candidate_info(**kwargs):
        return kwargs

    return StructuredTool.from_function(
        func=save_candidate_info,
        name="save_candidate_info",
        description=(
            "Save the candidate information you actually heard. "
            "Leave a field null if the candidate did not mention "
            "it — never guess or invent a value."
        ),
        args_schema=ExtractionModel,
    )


def _clean_updates(
    args: dict,
    fields: list[dict],
) -> dict:
    """
    Keep only real values for the fields we asked about. Drops
    nulls, empty strings, empty lists, and anything the model
    hallucinated outside the allowed field set.
    """

    allowed = {field["name"] for field in fields}

    updates = {}

    for name, value in args.items():

        if name not in allowed:
            continue

        if value is None:
            continue

        if isinstance(value, str):

            value = value.strip()

            if not value:
                continue

        elif isinstance(value, list):

            cleaned = [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]

            if not cleaned:
                continue

            value = cleaned

        updates[name] = value

    return updates


def _describe_fields(fields: list[dict]) -> str:

    lines = []

    for field in fields:

        kind = "list of strings" if field.get("multi") else "string"

        lines.append(
            f"- {field['name']} ({kind}): {field['description']}"
        )

    return "\n".join(lines)


# ============================================================
# Prompts
# ============================================================

BULK_PROMPT = """
You extract structured information from an Egyptian Arabic
candidate's spoken self-introduction in a recruitment interview.

The candidate was asked to say everything about themselves at
once, so their answer may contain many pieces of information, in
any order, or only a few.

Call save_candidate_info exactly once.

RULES:

1. Fill in ONLY what the candidate actually said.
2. Leave every other field null. Missing information is normal and
   expected — it will be asked about separately afterwards.
3. NEVER guess, infer or invent a value. If you are not sure, leave
   it null.
4. "لسه متخرج" or "ممعنديش خبرة" means years_of_experience = "0".
   Zero experience is a real answer, not a missing one.
5. For list fields, put each skill/tool as its own separate item.
6. The speech comes from an imperfect speech-to-text system, so
   minor noise is normal. Extract what is clearly there and ignore
   the rest.

FIELDS:

{fields}

CANDIDATE INTRODUCTION:

{transcript}
"""


SINGLE_PROMPT = """
You are validating and extracting the answer to ONE specific
question in an Egyptian Arabic recruitment interview.

QUESTION ASKED:

{question}

FIELD TO FILL:

{fields}

RULES:

1. If the candidate answered THIS question, call
   save_candidate_info with the value.
2. If they did NOT answer this question — they greeted you, said
   an acknowledgement like "تمام" or "أيوه", asked a question,
   refused, said something unrelated, or answered a DIFFERENT
   question — do NOT call the tool at all.
3. NEVER guess or invent a value.
4. "لسه متخرج" or "ممعنديش خبرة" means years_of_experience = "0".
   Zero experience is a valid answer — call the tool.
5. For list fields, put each skill/tool as its own separate item.

CANDIDATE ANSWER:

{transcript}
"""


# ============================================================
# Extraction calls
# ============================================================

async def _run(
    prompt: str,
    transcript: str,
    fields: list[dict],
) -> dict:

    tool = _build_tool(fields)

    extraction_llm = llm.bind_tools([tool])

    response = await extraction_llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=transcript),
        ]
    )

    print("[EXTRACTION] Tool calls:", response.tool_calls)

    if not response.tool_calls:
        return {}

    updates = {}

    for tool_call in response.tool_calls:

        if tool_call["name"] != "save_candidate_info":
            continue

        updates.update(
            _clean_updates(
                tool_call.get("args", {}),
                fields,
            )
        )

    return updates


async def extract_all(
    transcript: str,
    fields: list[dict],
) -> dict:
    """
    One call over the whole opening paragraph, returning every
    field the candidate mentioned. Returns {} if nothing usable was
    found — that's not an error here, it just means every required
    field gets asked individually afterwards.
    """

    print("[EXTRACTION] Bulk extraction from intro paragraph")

    prompt = BULK_PROMPT.format(
        fields=_describe_fields(fields),
        transcript=transcript,
    )

    updates = await _run(prompt, transcript, fields)

    print("[EXTRACTION] Extracted:", updates)

    return updates


async def extract_one(
    transcript: str,
    field: dict,
) -> dict:
    """
    Targeted extraction for a single follow-up question. Returns
    {} when the candidate didn't actually answer it.
    """

    print("[EXTRACTION] Single-field extraction:", field["name"])

    prompt = SINGLE_PROMPT.format(
        question=field["question"],
        fields=_describe_fields([field]),
        transcript=transcript,
    )

    updates = await _run(prompt, transcript, [field])

    print("[EXTRACTION] Extracted:", updates)

    return updates
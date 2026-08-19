"""
Test the LLM-driven HR information extraction.

This test simulates an Egyptian Arabic recruitment conversation and
checks whether the Groq LLM correctly uses the extraction tool to
populate the structured candidate record.

It does NOT test:
- LiveKit
- STT
- TTS
- Database
- Mem0

It tests only:
    Conversation -> Groq LLM -> function tool -> CallExtraction
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.extraction import CallExtraction

load_dotenv()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file before running this test."
    )


# Groq provides an OpenAI-compatible API.
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------
# Extraction tool schema
# ---------------------------------------------------------------------

extraction = CallExtraction()


tools = [
    {
        "type": "function",
        "function": {
            "name": "update_candidate_info",
            "description": (
                "Save HR information explicitly provided by the candidate "
                "during the conversation. Only save information the "
                "candidate actually stated. Never guess or invent missing "
                "information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_name": {
                        "type": ["string", "null"],
                        "description": "Candidate's full name.",
                    },
                    "contact_info": {
                        "type": ["string", "null"],
                        "description": (
                            "Candidate's phone number or email address."
                        ),
                    },
                    "position": {
                        "type": ["string", "null"],
                        "description": (
                            "The job position the candidate is applying for."
                        ),
                    },
                    "experience": {
                        "type": ["string", "null"],
                        "description": (
                            "Candidate's years of experience and "
                            "relevant qualifications."
                        ),
                    },
                    "current_salary": {
                        "type": ["string", "null"],
                        "description": "Candidate's current salary.",
                    },
                    "expected_salary": {
                        "type": ["string", "null"],
                        "description": "Candidate's expected salary.",
                    },
                    "availability": {
                        "type": ["string", "null"],
                        "description": (
                            "When the candidate can start working, "
                            "including notice period."
                        ),
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": (
                            "Other relevant information explicitly "
                            "provided by the candidate."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]


# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
أنت موظف في قسم الموارد البشرية في شركة.

أنت بتتكلم باللهجة المصرية بشكل طبيعي ومهذب ومحترف.

أثناء المحادثة، هدفك جمع معلومات المتقدم للوظيفة.

المعلومات المطلوبة:

- الاسم بالكامل
- بيانات التواصل
- الوظيفة اللي بيتقدملها
- سنين الخبرة والمؤهلات
- المرتب الحالي
- المرتب المتوقع
- إمتى يقدر يبدأ الشغل
- أي ملاحظات إضافية مهمة

مهم جداً:

1. لما المتصل يذكر معلومة تخص بياناته، استخدم أداة
   update_candidate_info فوراً لتسجيل المعلومة.

2. سجل فقط المعلومات التي قالها المتصل بوضوح.

3. ممنوع تخمين أو اختراع أي معلومة.

4. لو المتصل ذكر أكثر من معلومة في نفس الجملة، سجل كل المعلومات
   التي ذكرها في نفس استدعاء للأداة.

5. لو المتصل صحح معلومة سابقة، استخدم القيمة الجديدة.

6. لا تطلب من المتصل إعادة معلومة قام بذكرها بالفعل.

7. لا تسجل كلام موظف الموارد البشرية على أنه معلومات للمتقدم.

8. لا تسجل الأسئلة التي يطرحها المتصل على أنها بيانات شخصية.

9. لا تستخدم أداة extraction إلا عندما يقدم المتصل معلومة فعلية.

10. بعد استخدام الأداة، كمل المحادثة بشكل طبيعي.

المحادثة التي أمامك هي محادثة افتراضية لغرض اختبار استخراج البيانات.
"""


# ---------------------------------------------------------------------
# Simulated conversation
# ---------------------------------------------------------------------

conversation = [
    {
        "role": "user",
        "content": "مساء الخير، أنا أحمد محمد.",
    },
    {
        "role": "assistant",
        "content": "أهلاً يا أستاذ أحمد، ممكن أعرف حضرتك بتقدم على وظيفة إيه؟",
    },
    {
        "role": "user",
        "content": "أنا بقدم على وظيفة مهندس صيانة، وعندي حوالي 3 سنين خبرة في المجال.",
    },
    {
        "role": "assistant",
        "content": "تمام. ممكن أعرف مرتب حضرتك الحالي؟",
    },
    {
        "role": "user",
        "content": "مرتبي الحالي حوالي 15 ألف جنيه.",
    },
    {
        "role": "assistant",
        "content": "تمام، والمرتب المتوقع يكون كام؟",
    },
    {
        "role": "user",
        "content": "متوقع حوالي 20 ألف.",
    },
    {
        "role": "assistant",
        "content": "تمام. ولو حصل قبول، تقدر تبدأ الشغل إمتى؟",
    },
    {
        "role": "user",
        "content": "أقدر أبدأ بعد شهر.",
    },
]


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LLM EXTRACTION TEST")
    print("=" * 70)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for index, message in enumerate(conversation, start=1):

        print(f"\n{'-' * 70}")
        print(f"TURN {index}")
        print(f"{message['role'].upper()}: {message['content']}")

        messages.append(message)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # -------------------------------------------------------------
        # Check whether the model called our extraction tool
        # -------------------------------------------------------------

        if assistant_message.tool_calls:

            for tool_call in assistant_message.tool_calls:

                if tool_call.function.name != "update_candidate_info":
                    continue

                import json

                arguments = json.loads(
                    tool_call.function.arguments
                )

                print("\n[TOOL CALL]")
                print(arguments)

                # Update our real extraction state.
                extraction.update(**arguments)

                print("\n[CURRENT RECORD]")
                print(extraction.get_record())

                # Add the assistant's tool-call message to the
                # conversation history.
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        ],
                    }
                )

                # Tell the model that the tool succeeded.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "تم حفظ المعلومات بنجاح.",
                    }
                )

        else:
            print("\n[NO TOOL CALL]")

        # -------------------------------------------------------------
        # Show normal assistant response if there is one.
        # -------------------------------------------------------------

        if assistant_message.content:
            print("\n[LLM RESPONSE]")
            print(assistant_message.content)

    # -----------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EXTRACTED RECORD")
    print("=" * 70)

    record = extraction.get_record()

    for field, value in record.items():
        print(f"{field:20} : {value}")

    print("\nMissing fields:")
    print(extraction.get_missing_fields())

    print("\nExtraction successful:")
    print(extraction.has_information())


if __name__ == "__main__":
    main()
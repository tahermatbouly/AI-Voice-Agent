from agent.memory import CallerMemory


CALLER_ID = "test-caller-001"


def main():
    print("=" * 70)
    print("MEM0 MEMORY TEST")
    print("=" * 70)

    memory = CallerMemory()

    print("\nMem0 available:")
    print(memory.available)

    if not memory.available:
        print("\nMem0 could not be initialized.")
        print("Check the installation/error message above.")
        return

    # ---------------------------------------------------------------
    # First call
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("SAVING FIRST CALL")
    print("-" * 70)

    conversation = [
        {
            "role": "user",
            "content": "أنا أحمد محمد، بقدم على وظيفة مهندس صيانة.",
        },
        {
            "role": "assistant",
            "content": "تمام يا أستاذ أحمد، عندك خبرة قد إيه؟",
        },
        {
            "role": "user",
            "content": "عندي حوالي 3 سنين خبرة في صيانة المعدات.",
        },
    ]

    success = memory.save_conversation(
        caller_id=CALLER_ID,
        conversation=conversation,
    )

    print("Save successful:", success)

    # ---------------------------------------------------------------
    # Search memory
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("SEARCHING MEMORY")
    print("-" * 70)

    memories = memory.get_memories(
        caller_id=CALLER_ID,
        query="What do we know about this candidate?",
    )

    if memories:
        for index, item in enumerate(memories, start=1):
            print(f"{index}. {item}")
    else:
        print("No memories found.")

    # ---------------------------------------------------------------
    # Save structured record
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("SAVING STRUCTURED RECORD")
    print("-" * 70)

    record = {
        "candidate_name": "أحمد محمد",
        "contact_info": "",
        "position": "مهندس صيانة",
        "experience": "3 سنين خبرة في صيانة المعدات",
        "current_salary": "15000",
        "expected_salary": "20000",
        "availability": "بعد شهر",
        "notes": "",
    }

    success = memory.save_record(
        caller_id=CALLER_ID,
        record=record,
    )

    print("Save successful:", success)

    # ---------------------------------------------------------------
    # Search again
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("SEARCHING MEMORY AGAIN")
    print("-" * 70)

    memories = memory.get_memories(
        caller_id=CALLER_ID,
        query="What is this candidate's name, job position and experience?",
    )

    if memories:
        for index, item in enumerate(memories, start=1):
            print(f"{index}. {item}")
    else:
        print("No memories found.")


if __name__ == "__main__":
    main()
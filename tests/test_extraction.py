from agent.extraction import CallExtraction


def main():
    extraction = CallExtraction()

    print("Initial record:")
    print(extraction.get_record())

    extraction.update(
        candidate_name="أحمد محمد",
        experience="3 سنين خبرة في المبيعات",
    )

    print("\nAfter first update:")
    print(extraction.get_record())

    extraction.update(
        position="مهندس صيانة",
        current_salary="15000",
        expected_salary="20000",
    )

    print("\nAfter second update:")
    print(extraction.get_record())

    extraction.update(
        availability="بعد شهر",
    )

    print("\nFinal record:")
    print(extraction.get_record())

    print("\nMissing fields:")
    print(extraction.get_missing_fields())

    print("\nHas information:")
    print(extraction.has_information())


if __name__ == "__main__":
    main()
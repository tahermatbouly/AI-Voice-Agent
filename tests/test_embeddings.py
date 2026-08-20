import time

from agent.embeddings import EmbeddingModel


def main():
    print("=" * 70)
    print("EMBEDDING MODEL TEST")
    print("=" * 70)

    print("\nLoading embedding model...")

    start = time.perf_counter()

    model = EmbeddingModel()

    load_time = time.perf_counter() - start

    print(f"Model loaded in: {load_time:.3f} seconds")
    print(f"Model dimension: {model.dimension}")

    sentences = [
        "أحمد محمد مهندس صيانة عنده 3 سنين خبرة.",
        "المتقدم اسمه أحمد محمد وبيقدم على وظيفة مهندس صيانة.",
        "عنده خبرة ثلاث سنوات في مجال صيانة المعدات.",
        "المرتب الحالي 15 ألف جنيه والمرتب المتوقع 20 ألف.",
        "يقدر يبدأ الشغل بعد شهر.",
    ]

    print("\n" + "-" * 70)
    print("ENCODING TEST")
    print("-" * 70)

    for sentence in sentences:
        start = time.perf_counter()

        embedding = model.encode(sentence)

        elapsed = time.perf_counter() - start

        print(f"\nText: {sentence}")
        print(f"Time: {elapsed * 1000:.2f} ms")
        print(f"Dimensions: {len(embedding)}")

    print("\n" + "-" * 70)
    print("BATCH TEST")
    print("-" * 70)

    start = time.perf_counter()

    embeddings = model.encode_many(sentences)

    elapsed = time.perf_counter() - start

    print(f"Texts: {len(sentences)}")
    print(f"Total time: {elapsed * 1000:.2f} ms")
    print(
        f"Average per text: "
        f"{(elapsed / len(sentences)) * 1000:.2f} ms"
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
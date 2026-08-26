from .egyptian_dictionary import EGYPTIAN_DICTIONARY


def normalize_egyptian(text: str) -> str:
    words = text.split()

    normalized = []

    for word in words:
        normalized.append(
            EGYPTIAN_DICTIONARY.get(word, word)
        )

    return " ".join(normalized)
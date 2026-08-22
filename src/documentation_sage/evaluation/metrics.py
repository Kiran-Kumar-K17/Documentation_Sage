from typing import Any


def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Calculate Recall@K.

    Returns 1.0 if at least one expected source
    appears in the retrieved sources.
    """

    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)

    if not expected_set:
        return 0.0

    hits = retrieved_set.intersection(expected_set)

    return len(hits) / len(expected_set)


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Finds the rank of the first correct source.
    """

    expected_set = set(expected_sources)

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source in expected_set:
            return 1.0 / rank

    return 0.0


def calculate_average(
    values: list[float],
) -> float:
    """
    Calculate the average of a list.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)

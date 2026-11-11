from collections import defaultdict

import pandas
import pytest

from deck_generation.bin.config import SentenceFilteringConfig
from deck_generation.data_generation.sentence_filterer import SentenceFilterer

# TODO: Test with too many dissimilar sentences (5 very different sentences, 2 must be kept)
# TODO: Basic usage test (3 sentences to keep, 2 very different and 2 quite close)
# TODO: Test that even with only very similar sentences it returns the expected number of sentences


@pytest.mark.parametrize(
    "input_sentence_words, expected_kept_sentences_indexes, group_word, max_sentences_count_per_new_word, max_sentence_similarity_per_new_word",
    [
        pytest.param(
            [
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "something", "to", "eat"],
                ["Something", "terrible", "just", "happened"],
                ["Isn't", "that", "the", "great", "something", "ever"],
            ],
            [1, 2, 3],
            "something",
            3,
            0.0,
            id="it_keeps_one_of_the_two_similar_sentences",
        )
    ],
)
def test_it_produces_the_expected_sentences_when_filtering_groups(
    input_sentence_words: list[list[str]],
    expected_kept_sentences_indexes: list[int],
    group_word: str,
    max_sentences_count_per_new_word: int,
    max_sentence_similarity_per_new_word: float,
) -> None:
    config = SentenceFilteringConfig(
        max_sentences_count_per_new_word=max_sentences_count_per_new_word,
        max_sentence_similarity_per_new_word=max_sentence_similarity_per_new_word,
    )

    dummy_loader = lambda: pandas.DataFrame()
    filterer = SentenceFilterer(
        sentences_loader=dummy_loader,
        word_to_freq=defaultdict(lambda: 1),
        config=config,
    )
    sentences_group_words = pandas.Series(input_sentence_words)
    filtered_sentences_indices = filterer._get_filtered_similar_sentences_indices(
        sentences_group_words=sentences_group_words, group_word=group_word
    )

    assert len(filtered_sentences_indices) == max_sentences_count_per_new_word
    assert set(filtered_sentences_indices) == set(expected_kept_sentences_indexes)

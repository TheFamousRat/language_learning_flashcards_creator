from collections import defaultdict

import pandas
import pytest

from deck_generation.deck_generation.config import SentenceFilteringConfig
from deck_generation.preprocessing.sentence_filterer import SentenceFilterer


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
        ),
        pytest.param(
            [
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "to", "eat", "something"],
            ],
            [2, 3, 4],
            "something",
            3,
            0.0,
            id="it_keeps_the_correct_number_of_sentences_despite_high_similarity",
        ),
        pytest.param(
            [
                ["something", "is", "terribly", "wrong", "here"],
                ["aren't", "you", "something"],
                ["has", "something", "good", "happened"],
                ["there's", "something", "in", "there"],
                ["I", "want", "to", "eat", "something"],
            ],
            [0, 4],
            "something",
            2,
            0.25,
            id="it_keeps_the_correct_number_of_sentences_despite_low_similarity",
        ),
        pytest.param(
            [
                ["something", "is", "terribly", "wrong", "here"],
                ["aren't", "you", "something"],
                ["has", "something", "good", "happened"],
                ["there's", "something", "in", "there"],
                ["I", "want", "to", "eat", "something"],
                ["I", "want", "something", "to", "eat"],
            ],
            [0, 5],
            "something",
            2,
            0.25,
            id="it_keeps_the_correct_number_of_sentences_despite_too_many_dissimilar_sentences",
        ),
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
    unknown_indices = set(filtered_sentences_indices).difference(
        set(sentences_group_words.index)
    )
    assert len(unknown_indices) == 0
    assert set(filtered_sentences_indices) == set(expected_kept_sentences_indexes)

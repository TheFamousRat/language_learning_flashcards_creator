from __future__ import annotations
import pytest
import numpy as np

from tests.dummy_word_frequencies import (
    CHINESE_WORD_FREQUENCIES,
    ENGLISH_WORD_FREQUENCIES,
    FRENCH_WORD_FREQUENCIES,
    GERMAN_WORD_FREQUENCIES,
)


import re

LETTER_RUN_PATTERN = re.compile(r"[^\W\d_]+")


def segment_sentence(
    sentence: str, token_frequencies: dict[str, int]
) -> list[tuple[int, int, str]]:
    total_log_frequency = np.log(sum(token_frequencies.values()))
    token_to_log_probability = {
        token: np.log(frequency) - total_log_frequency
        for token, frequency in token_frequencies.items()
    }
    longest_token_length = max(map(len, token_frequencies))

    sentence_length = len(sentence)
    best_scores: list[tuple[int, float]] = [(0, 0.0)] * (sentence_length + 1)
    best_starts: list[int] = [0] * (sentence_length + 1)
    best_tokens: list[str | None] = [None] * (sentence_length + 1)

    for end in range(1, sentence_length + 1):
        best_scores[end] = best_scores[end - 1]
        best_starts[end] = end - 1

        for start in range(max(0, end - longest_token_length), end):
            substring = sentence[start:end]
            possible_tokens = (
                [substring, substring[0].lower() + substring[1:]]
                if start == 0
                else [substring]
            )
            candidate_tokens = [
                token for token in possible_tokens if token in token_to_log_probability
            ]
            if not candidate_tokens:
                continue

            token = max(candidate_tokens, key=token_to_log_probability.__getitem__)
            explained_characters, log_probability = best_scores[start]
            score = (
                explained_characters + end - start,
                log_probability + token_to_log_probability[token],
            )

            if score > best_scores[end]:
                best_scores[end] = score
                best_starts[end] = start
                best_tokens[end] = token

    token_spans: list[tuple[int, int, str]] = []
    position = sentence_length
    while position > 0:
        token = best_tokens[position]
        start = best_starts[position]
        if token is not None:
            token_spans.append((start, position, token))
        position = start

    return token_spans[::-1]


def tokenize_sentence(sentence: str, token_frequencies: dict[str, int]) -> list[str]:
    token_spans = segment_sentence(sentence, token_frequencies)

    explained_positions = {
        position for start, end, _ in token_spans for position in range(start, end)
    }

    positions_inside_names: set[int] = set()
    for word_match in LETTER_RUN_PATTERN.finditer(sentence):
        word_positions = range(word_match.start(), word_match.end())
        is_capitalized = word_match.group()[0].isupper()
        is_fully_explained = all(
            position in explained_positions for position in word_positions
        )
        if is_capitalized and not is_fully_explained:
            positions_inside_names.update(word_positions)

    return [
        token for start, _, token in token_spans if start not in positions_inside_names
    ]


@pytest.mark.parametrize(
    "sentence, word_frequencies, expected_tokens",
    [
        pytest.param(
            "Je ne sais pas.",
            FRENCH_WORD_FREQUENCIES,
            ["je", "ne", "sais", "pas"],
            id="simple_french",
        ),
        pytest.param(
            "It is cheap.",
            ENGLISH_WORD_FREQUENCIES,
            ["it", "is", "cheap"],
            id="simple_english",
        ),
        pytest.param(
            "Das ist gut.",
            GERMAN_WORD_FREQUENCIES,
            ["das", "ist", "gut"],
            id="simple_german",
        ),
        pytest.param(
            "我们今天去学校。",
            CHINESE_WORD_FREQUENCIES,
            ["我们", "今天", "去", "学校"],
            id="simple_chinese",
        ),
        pytest.param(
            "Alors je mange.",
            FRENCH_WORD_FREQUENCIES,
            ["alors", "je", "mange"],
            id="initial_capital_french_first_word_is_lowercased",
        ),
        pytest.param(
            "Pain et fromage sont bons ici.",
            FRENCH_WORD_FREQUENCIES,
            ["pain", "et", "fromage", "sont", "bons", "ici"],
            id="initial_capital_french_first_noun_is_lowercased",
        ),
        pytest.param(
            "The cat sleeps on the sofa.",
            ENGLISH_WORD_FREQUENCIES,
            ["the", "cat", "sleeps", "on", "the", "sofa"],
            id="initial_capital_english_first_word_is_lowercased",
        ),
        pytest.param(
            "Bread and water are cheap here.",
            ENGLISH_WORD_FREQUENCIES,
            ["bread", "and", "water", "are", "cheap", "here"],
            id="initial_capital_english_first_noun_is_lowercased",
        ),
        pytest.param(
            "I like cats.",
            ENGLISH_WORD_FREQUENCIES,
            ["I", "like", "cats"],
            id="initial_capital_english_pronoun_keeps_its_capital",
        ),
        pytest.param(
            "Ich esse gern Brot.",
            GERMAN_WORD_FREQUENCIES,
            ["ich", "esse", "gern", "Brot"],
            id="initial_capital_german_first_word_takes_most_frequent_variant",
        ),
        pytest.param(
            "Brot ist gut.",
            GERMAN_WORD_FREQUENCIES,
            ["Brot", "ist", "gut"],
            id="initial_capital_german_first_noun_keeps_its_capital",
        ),
        pytest.param(
            "Das Essen ist gut.",
            GERMAN_WORD_FREQUENCIES,
            ["das", "Essen", "ist", "gut"],
            id="initial_capital_german_noun_after_first_word_keeps_its_capital",
        ),
        pytest.param(
            "问题是我们今天去学校。",
            CHINESE_WORD_FREQUENCIES,
            ["问题", "是", "我们", "今天", "去", "学校"],
            id="initial_capital_chinese_has_no_case_to_normalize",
        ),
        pytest.param(
            "Alors, l'homme mange.",
            FRENCH_WORD_FREQUENCIES,
            ["alors", "homme", "mange"],
            id="unknown_chunk_french_elision_drops_unknown_article",
        ),
        pytest.param(
            "Aujourd'hui, je ne sais pas.",
            FRENCH_WORD_FREQUENCIES,
            ["aujourd'hui", "je", "ne", "sais", "pas"],
            id="unknown_chunk_french_known_apostrophe_word_is_kept_whole",
        ),
        pytest.param(
            "You mustn't lose it.",
            ENGLISH_WORD_FREQUENCIES,
            ["you", "must", "lose", "it"],
            id="unknown_chunk_english_contraction_keeps_known_stem",
        ),
        pytest.param(
            "I lost my notebook.",
            ENGLISH_WORD_FREQUENCIES,
            ["I", "lost", "my", "note", "book"],
            id="unknown_chunk_english_compound_is_split_into_known_parts",
        ),
        pytest.param(
            "Ich suche meinen Handschuh.",
            GERMAN_WORD_FREQUENCIES,
            ["ich", "suche", "meinen", "Handschuh"],
            id="unknown_chunk_german_known_compound_is_kept_whole",
        ),
        pytest.param(
            "Meine E-Mail-Adresse ist neu.",
            GERMAN_WORD_FREQUENCIES,
            ["meine", "E-Mail", "Adresse", "ist", "neu"],
            id="unknown_chunk_german_hyphenated_compound_is_split_on_hyphens",
        ),
        pytest.param(
            "他说，我喜欢吃中国菜。",
            CHINESE_WORD_FREQUENCIES,
            ["他", "说", "我", "喜欢", "吃", "中国菜"],
            id="unknown_chunk_chinese_known_compound_beats_its_parts",
        ),
        pytest.param(
            "研究生命。",
            CHINESE_WORD_FREQUENCIES,
            ["研究", "生命"],
            id="unknown_chunk_chinese_most_probable_split_beats_longest_match",
        ),
        pytest.param(
            "Marie a vu l'homme.",
            FRENCH_WORD_FREQUENCIES,
            ["a", "vu", "homme"],
            id="proper_noun_french_first_name_is_dropped",
        ),
        pytest.param(
            "Hier, Jean-Pierre et Léa ont mangé du pain.",
            FRENCH_WORD_FREQUENCIES,
            ["hier", "et", "ont", "mangé", "du", "pain"],
            id="proper_noun_french_hyphenated_and_accented_names_are_dropped",
        ),
        pytest.param(
            "Mary and Tom lost their notebook in Paris.",
            ENGLISH_WORD_FREQUENCIES,
            ["and", "lost", "their", "note", "book", "in"],
            id="proper_noun_english_names_and_city_are_dropped",
        ),
        pytest.param(
            "Anna und Hans essen gern Brot.",
            GERMAN_WORD_FREQUENCIES,
            ["und", "essen", "gern", "Brot"],
            id="proper_noun_german_names_are_dropped_and_nouns_are_kept",
        ),
        pytest.param(
            "今天，王芳和李明去学校。",
            CHINESE_WORD_FREQUENCIES,
            ["今天", "和", "去", "学校"],
            id="proper_noun_chinese_names_of_unknown_characters_are_dropped",
        ),
        pytest.param(
            "Aujourd'hui, je ne sais pas si l'homme que j'ai vu hier mange encore ici.",
            FRENCH_WORD_FREQUENCIES,
            [
                "aujourd'hui",
                "je",
                "ne",
                "sais",
                "pas",
                "si",
                "homme",
                "que",
                "ai",
                "vu",
                "hier",
                "mange",
                "encore",
                "ici",
            ],
            id="long_french_several_elisions",
        ),
        pytest.param(
            "Pain et fromage ne coûtent pas cher ici, mais l'homme mange du pain avec du fromage.",
            FRENCH_WORD_FREQUENCIES,
            [
                "pain",
                "et",
                "fromage",
                "ne",
                "coûtent",
                "pas",
                "cher",
                "ici",
                "mais",
                "homme",
                "mange",
                "du",
                "pain",
                "avec",
                "du",
                "fromage",
            ],
            id="long_french_noun_start_with_elision_and_repeated_nouns",
        ),
        pytest.param(
            "Bread and water are cheap here, but the cat sleeps on the sofa.",
            ENGLISH_WORD_FREQUENCIES,
            [
                "bread",
                "and",
                "water",
                "are",
                "cheap",
                "here",
                "but",
                "the",
                "cat",
                "sleeps",
                "on",
                "the",
                "sofa",
            ],
            id="long_english_noun_start_with_two_clauses",
        ),
        pytest.param(
            "You mustn't lose your notebook, because it isn't cheap.",
            ENGLISH_WORD_FREQUENCIES,
            [
                "you",
                "must",
                "lose",
                "your",
                "note",
                "book",
                "because",
                "it",
                "isn't",
                "cheap",
            ],
            id="long_english_unknown_contraction_and_unknown_compound",
        ),
        pytest.param(
            "Mary and Tom lost their notebook in Paris, and it isn't cheap.",
            ENGLISH_WORD_FREQUENCIES,
            [
                "and",
                "lost",
                "their",
                "note",
                "book",
                "in",
                "and",
                "it",
                "isn't",
                "cheap",
            ],
            id="long_english_names_compound_and_contraction",
        ),
        pytest.param(
            "Brot und Wasser sind gut, aber der Handschuh ist neu.",
            GERMAN_WORD_FREQUENCIES,
            [
                "Brot",
                "und",
                "Wasser",
                "sind",
                "gut",
                "aber",
                "der",
                "Handschuh",
                "ist",
                "neu",
            ],
            id="long_german_noun_start_with_several_capitalized_nouns",
        ),
        pytest.param(
            "Meine E-Mail-Adresse ist neu, aber sie funktioniert nicht.",
            GERMAN_WORD_FREQUENCIES,
            [
                "meine",
                "E-Mail",
                "Adresse",
                "ist",
                "neu",
                "aber",
                "sie",
                "funktioniert",
                "nicht",
            ],
            id="long_german_hyphenated_compound",
        ),
        pytest.param(
            "Anna und Hans essen gern Brot, aber sie finden den Handschuh nicht.",
            GERMAN_WORD_FREQUENCIES,
            [
                "und",
                "essen",
                "gern",
                "Brot",
                "aber",
                "sie",
                "finden",
                "den",
                "Handschuh",
                "nicht",
            ],
            id="long_german_names_and_nouns",
        ),
        pytest.param(
            "研究生命的起源是一个有趣的问题。",
            CHINESE_WORD_FREQUENCIES,
            ["研究", "生命", "的", "起源", "是", "一个", "有趣", "的", "问题"],
            id="long_chinese_most_probable_split",
        ),
    ],
)
def test_it_tokenizes_sentences_from_a_frequency_dictionary_as_expected(
    sentence: str,
    word_frequencies: dict[str, int],
    expected_tokens: list[str],
) -> None:
    assert (
        tokenize_sentence(sentence=sentence, token_frequencies=word_frequencies)
        == expected_tokens
    )

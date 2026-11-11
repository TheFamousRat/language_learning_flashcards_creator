from __future__ import annotations

import re

import numpy as np
import regex


class ViterbiTokenizer:
    LETTER_RUN_PATTERN = re.compile(r"[^\W\d_]+")
    NOISY_CHARACTER_PATTERN = regex.compile(
        r"[^\p{L}\p{M}\p{N}'\-‐‑]|(?<!\p{L})[-‐‑]|[-‐‑](?!\p{L})"
    )
    SENTENCE_TERMINATORS = ".!?…:。！？："
    OPENING_CHARACTERS = "\"'«»“”„‘’‹›¿¡([{<-–—―"

    def __init__(
        self, token_to_log_probability: dict[str, float], longest_token_length: int
    ) -> None:
        self.token_to_log_probability = token_to_log_probability
        self.longest_token_length = longest_token_length

    @classmethod
    def from_token_frequencies(
        cls, token_frequencies: dict[str, int]
    ) -> ViterbiTokenizer:
        total_log_frequency = np.log(sum(token_frequencies.values()))
        token_to_log_probability = {
            token: np.log(frequency) - total_log_frequency
            for token, frequency in token_frequencies.items()
        }
        longest_token_length = max(map(len, token_frequencies))

        return ViterbiTokenizer(
            token_to_log_probability=token_to_log_probability,
            longest_token_length=longest_token_length,
        )

    def _find_sentence_initial_positions(self, sentence: str) -> set[int]:
        sentence_initial_positions: set[int] = set()

        for word_match in self.LETTER_RUN_PATTERN.finditer(sentence):
            position = word_match.start() - 1
            while position >= 0 and (
                sentence[position].isspace()
                or sentence[position] in self.OPENING_CHARACTERS
            ):
                position -= 1

            if position < 0 or sentence[position] in self.SENTENCE_TERMINATORS:
                sentence_initial_positions.add(word_match.start())

        return sentence_initial_positions

    def _segment_sentence(self, sentence: str) -> list[tuple[int, int, str]]:
        sentence_length = len(sentence)
        sentence_initial_positions = self._find_sentence_initial_positions(sentence)
        sentence = regex.sub(self.NOISY_CHARACTER_PATTERN, " ", sentence)

        best_scores: list[tuple[int, float]] = [(0, 0.0)] * (sentence_length + 1)
        best_starts: list[int] = [0] * (sentence_length + 1)
        best_tokens: list[str | None] = [None] * (sentence_length + 1)
        for end in range(1, sentence_length + 1):
            best_scores[end] = best_scores[end - 1]
            best_starts[end] = end - 1

            for start in range(max(0, end - self.longest_token_length), end):
                substring = sentence[start:end]
                possible_tokens = (
                    [substring, substring[0].lower() + substring[1:]]
                    if start in sentence_initial_positions
                    else [substring]
                )
                candidate_tokens = [
                    token
                    for token in possible_tokens
                    if token in self.token_to_log_probability
                ]
                if not candidate_tokens:
                    continue

                token = max(
                    candidate_tokens, key=self.token_to_log_probability.__getitem__
                )
                explained_characters, log_probability = best_scores[start]
                score = (
                    explained_characters + end - start,
                    log_probability + self.token_to_log_probability[token],
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

    def tokenize_sentence(self, sentence: str) -> list[str]:
        token_spans = self._segment_sentence(sentence=sentence)

        explained_positions = {
            position for start, end, _ in token_spans for position in range(start, end)
        }

        positions_inside_names: set[int] = set()
        for word_match in self.LETTER_RUN_PATTERN.finditer(sentence):
            word_positions = range(word_match.start(), word_match.end())
            is_capitalized = word_match.group()[0].isupper()
            is_fully_explained = all(
                position in explained_positions for position in word_positions
            )
            if is_capitalized and not is_fully_explained:
                positions_inside_names.update(word_positions)

        return [
            token
            for start, _, token in token_spans
            if start not in positions_inside_names
        ]

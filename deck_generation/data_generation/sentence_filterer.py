from __future__ import annotations
from typing import Callable


import numpy as np
import pandas


from pathlib import Path

from deck_generation.bin.config import SentenceFilteringConfig
from deck_generation.constants import (
    TARGET_ID_COL_NAME,
    TARGET_SENTENCE_COL_NAME,
    TRANSLATED_ID_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
    SENTENCES_WORDS_COL_NAME,
    RAREST_WORD_COL_NAME,
    RAREST_WORD_FREQ_COL_NAME,
)
import regex
import logging

from deck_generation.utils import split_string_at_indices

_logger = logging.getLogger(name=__file__)
_logger.setLevel(level=logging.DEBUG)


class SentenceFilterer:
    def __init__(
        self,
        sentences_loader: Callable[..., pandas.DataFrame],
        word_to_freq: dict[str, int],
    ) -> None:
        self._sentences_loader = sentences_loader
        self._original_sentences_df: pandas.DataFrame | None = None
        self.word_to_freq = word_to_freq

    @classmethod
    def from_tatoeba_file(
        cls,
        sentences_filepath: Path,
        word_frequency_file_path: Path,
    ) -> SentenceFilterer:
        def _tatoeba_loader(
            _sentences_path: Path = sentences_filepath,
        ) -> pandas.DataFrame:
            _logger.info(
                f"Loading sentences from Tatoeba export file at {sentences_filepath}..."
            )
            loaded_sentences = pandas.read_csv(
                filepath_or_buffer=_sentences_path,
                sep="	",
                names=[
                    TARGET_ID_COL_NAME,
                    TARGET_SENTENCE_COL_NAME,
                    TRANSLATED_ID_COL_NAME,
                    TRANSLATED_SENTENCE_COL_NAME,
                ],
            )
            _logger.info(f"Loaded {len(loaded_sentences)} sentences.")

            return loaded_sentences

        word_freq_df = pandas.read_csv(
            filepath_or_buffer=word_frequency_file_path, sep=" ", names=["Word", "freq"]
        )

        word_to_freq: dict[str, int] = dict(
            zip(word_freq_df["Word"], word_freq_df["freq"])
        )

        return SentenceFilterer(
            sentences_loader=_tatoeba_loader,
            word_to_freq=word_to_freq,
        )

    @property
    def original_sentences_df(self) -> pandas.DataFrame:
        if self._original_sentences_df is None:
            self._original_sentences_df = self._sentences_loader()

        return self._original_sentences_df

    def _get_rarest_word_frequency(
        self, sentences_words: list[list[str]], only_proper_nouns_capitalized: bool
    ):
        # We assume that non-word starting words are capitalized properly
        non_starting_words = [
            sentence_word
            for sentence_words in sentences_words
            for sentence_word in sentence_words[1:]
        ]
        non_starting_word_frequencies: list[int | float] = [
            self.word_to_freq.get(word, -1) for word in non_starting_words
        ]

        if only_proper_nouns_capitalized:
            non_starting_word_frequencies = [
                np.inf if word[0].upper() == word[0] else freq
                for word, freq in zip(non_starting_words, non_starting_word_frequencies)
            ]

        starting_words = [
            (
                sentence_words[0]
                if sentence_words[0] in self.word_to_freq
                else sentence_words[0].lower()
            )
            for sentence_words in sentences_words
        ]
        starting_word_frequencies = [
            self.word_to_freq.get(word, np.inf) for word in starting_words
        ]

        word_frequencies = starting_word_frequencies + non_starting_word_frequencies
        words = starting_words + non_starting_words
        rarest_word_idx = np.argmin(word_frequencies)

        return words[rarest_word_idx], word_frequencies[rarest_word_idx]

    def _get_split_sentences_and_words(
        self, sentences_df: pandas.DataFrame
    ) -> pandas.Series:
        # Splits entries into sentences, and split these sentences into words
        sentence_first_letter_pattern = r"(?<=[^\p{L}|\s|\,|\;|\'|\d|\-]|^)\s*(\p{Lu})"
        split_sentences = sentences_df[TARGET_SENTENCE_COL_NAME].apply(
            lambda sentence: split_string_at_indices(
                string_to_split=sentence,
                split_indices=[
                    match.span()[0]
                    for match in regex.finditer(
                        pattern=sentence_first_letter_pattern, string=sentence
                    )
                ],
            )
        )
        sentences_words = split_sentences.apply(
            lambda sentences_list: [
                regex.findall(r"(\p{L}+\'?)", sentence) for sentence in sentences_list
            ]
        )

        return sentences_words

    def get_filtered_sentences_df(
        self,
        config: SentenceFilteringConfig,
    ) -> pandas.DataFrame:
        # TODO: Eventually a metric to check how similar two sentences are would be useful
        # This similarity would not be semantic, as learning two ways to say the same thing is useful, but instead
        # a sort of Levenshtein distance to measure the number of required letter changes between two sentences
        _logger.info("Running sentence filtering.")

        _logger.info("   Removing duplicate entries...")
        sentences_df = self.original_sentences_df.drop_duplicates(
            subset=TARGET_SENTENCE_COL_NAME
        )

        if TRANSLATED_SENTENCE_COL_NAME in sentences_df.columns:
            sentences_df = sentences_df.drop_duplicates(
                subset=TRANSLATED_SENTENCE_COL_NAME
            )

        _logger.info("   Splitting entries into separate sentences...")
        sentences_df[SENTENCES_WORDS_COL_NAME] = self._get_split_sentences_and_words(
            sentences_df=sentences_df
        )

        _logger.info("   Removing entries out of word count range...")
        sentences_lengths = sentences_df[SENTENCES_WORDS_COL_NAME].apply(
            lambda sentences_list: sum(
                [len(sentences_words) for sentences_words in sentences_list]
            )
        )
        sentences_length_mask = (sentences_lengths >= config.min_word_count) & (
            sentences_lengths <= config.max_word_count
        )
        sentences_df = sentences_df[sentences_length_mask]
        sentences_df = sentences_df.reset_index(drop=True)

        _logger.info("   Registering least frequent word per sentence...")
        rarest_word_and_frequency = sentences_df[SENTENCES_WORDS_COL_NAME].apply(
            lambda sentence: self._get_rarest_word_frequency(
                sentences_words=sentence,
                only_proper_nouns_capitalized=config.only_proper_nouns_capitalized,
            )
        )
        sentences_df[[RAREST_WORD_COL_NAME, RAREST_WORD_FREQ_COL_NAME]] = (
            pandas.DataFrame(rarest_word_and_frequency.to_list())
        )

        sentences_df = (
            sentences_df.groupby(RAREST_WORD_COL_NAME)
            .head(config.max_sentences_count_per_new_word)
            .sort_values(
                [RAREST_WORD_FREQ_COL_NAME, RAREST_WORD_COL_NAME], ascending=False
            )
        )

        _logger.info(
            f"Done, {len(sentences_df)} out of {len(self.original_sentences_df)} entries kept."
        )
        return sentences_df

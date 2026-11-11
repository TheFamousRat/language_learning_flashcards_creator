from __future__ import annotations

import logging
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas
import tqdm

from deck_generation.bin.config import SentenceFilteringConfig
from deck_generation.constants import (
    RAREST_WORD_COL_NAME,
    RAREST_WORD_FREQ_COL_NAME,
    SENTENCES_WORDS_COL_NAME,
    TARGET_ID_COL_NAME,
    TARGET_SENTENCE_COL_NAME,
    TRANSLATED_ID_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
)
from deck_generation.data_generation.viterbi_tokenizer import ViterbiTokenizer

_logger = logging.getLogger(name=__name__)
_logger.setLevel(level=logging.DEBUG)


class SentenceFilterer:
    def __init__(
        self,
        sentences_loader: Callable[..., pandas.DataFrame],
        word_to_freq: dict[str, int],
        config: SentenceFilteringConfig,
    ) -> None:
        self._sentences_loader = sentences_loader
        self.word_to_freq = word_to_freq
        self.config = config

    @classmethod
    def from_tatoeba_file(
        cls,
        sentences_filepath: Path,
        word_frequency_file_path: Path,
        config: SentenceFilteringConfig,
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
        word_freq_df["Word"] = word_freq_df["Word"].astype(str)
        word_freq_df["freq"] = word_freq_df["freq"].astype(int)
        word_freq_df = word_freq_df.dropna()

        word_to_freq: dict[str, int] = dict(
            zip(word_freq_df["Word"], word_freq_df["freq"])
        )

        return SentenceFilterer(
            sentences_loader=_tatoeba_loader,
            word_to_freq=word_to_freq,
            config=config,
        )

    @cached_property
    def original_sentences_df(self) -> pandas.DataFrame:
        return self._sentences_loader()

    def _get_filtered_similar_sentences_indices(
        self, sentences_group: pandas.DataFrame, group_word: str
    ) -> pandas.Index:
        group_size = len(sentences_group)
        kept_indices_mask = np.ones(shape=(group_size), dtype=bool)

        sentences_unique_words = sentences_group[SENTENCES_WORDS_COL_NAME].apply(
            lambda sentence: set(sentence).difference({group_word})
        )
        group_unique_words: set[str] = set()
        for words in sentences_group[SENTENCES_WORDS_COL_NAME]:
            group_unique_words.update(words)
        word_to_idx = {word: i for i, word in enumerate(group_unique_words)}
        weights_matrix = np.zeros(
            shape=(group_size, len(group_unique_words)), dtype=np.float32
        )
        for sentence_idx, sentence_words in enumerate(sentences_unique_words):
            for word in sentence_words:
                word_idx = word_to_idx[word]
                word_weight = 1.0 / np.sqrt(self.word_to_freq[word])
                weights_matrix[sentence_idx, word_idx] = word_weight

        intersection_mat = weights_matrix @ weights_matrix.T
        cardinality_mat = np.repeat(
            (weights_matrix**2.0).sum(axis=1)[None, :], group_size, axis=0
        )
        union_mat = cardinality_mat + cardinality_mat.T - intersection_mat
        sim_matrix = intersection_mat / union_mat

        np.fill_diagonal(
            sim_matrix, 0
        )  # TODO: Add config field so that some similarity is acceptable
        while kept_indices_mask.sum() > self.config.max_sentences_count_per_new_word:
            if sim_matrix.max() == 0.0:
                kept_indices_mask[
                    np.where(kept_indices_mask)[0][
                        self.config.max_sentences_count_per_new_word :
                    ]
                ] = False  # TODO: Maybe sort by eg sentence length before removing other sentences
                break

            index_to_remove = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)[0]
            sim_matrix[index_to_remove, :] = 0
            sim_matrix[:, index_to_remove] = 0
            kept_indices_mask[index_to_remove] = False

        kept_indices = sentences_group.index[kept_indices_mask]

        return kept_indices

    def _get_filtered_group_sentences_indices(
        self, sentences_df: pandas.DataFrame
    ) -> pandas.Index:
        # TODO: Test with too many dissimilar sentences (5 very different sentences, 2 must be kept)
        # TODO: Basic usage test (3 sentences to keep, 2 very different and 2 quite close)
        # TODO: Test that it behaves correctly with edge casey sentences count
        sentences_groups_by_rarest_word = sentences_df.groupby(RAREST_WORD_COL_NAME)
        filtered_sentences_indexes = pandas.Index([])
        for group_word, sentences_group in tqdm.tqdm(
            sentences_groups_by_rarest_word,
            desc="   Removing similar sentences in groups...",
        ):
            group_size = len(sentences_group)
            kept_indices = sentences_group.index

            if group_size > self.config.max_sentences_count_per_new_word:
                kept_indices = self._get_filtered_similar_sentences_indices(
                    sentences_group=sentences_group, group_word=group_word
                )

            filtered_sentences_indexes = filtered_sentences_indexes.append(kept_indices)

        return filtered_sentences_indexes

    def get_filtered_sentences_df(
        self,
    ) -> pandas.DataFrame:
        _logger.info("Running sentence filtering.")
        _logger.info("   Removing duplicate entries...")
        sentences_df = self.original_sentences_df.drop_duplicates(
            subset=TARGET_SENTENCE_COL_NAME
        )

        if TRANSLATED_SENTENCE_COL_NAME in sentences_df.columns:
            sentences_df = sentences_df.drop_duplicates(
                subset=TRANSLATED_SENTENCE_COL_NAME
            )

        _logger.info("   Tokenizing entries...")
        tokenizer = ViterbiTokenizer.from_token_frequencies(
            token_frequencies=self.word_to_freq
        )
        sentences_df[SENTENCES_WORDS_COL_NAME] = sentences_df[
            TARGET_SENTENCE_COL_NAME
        ].apply(lambda sentence: tokenizer.tokenize_sentence(sentence))

        _logger.info("   Removing entries out of word count range...")
        sentences_lengths = sentences_df[SENTENCES_WORDS_COL_NAME].apply(len)
        sentences_length_mask = (sentences_lengths >= self.config.min_word_count) & (
            sentences_lengths <= self.config.max_word_count
        )
        sentences_df = sentences_df[sentences_length_mask]

        _logger.info("   Ranking sentences by similarity...")
        rarest_word_and_frequency = sentences_df[SENTENCES_WORDS_COL_NAME].apply(
            lambda words_list: sorted(
                {word: self.word_to_freq[word] for word in words_list}.items(),
                key=lambda x: x[1],
            )[:2][
                -1
            ]  # TODO: Make 2 (meaning the 2nd rarest word) configurable
        )
        sentences_df[[RAREST_WORD_COL_NAME, RAREST_WORD_FREQ_COL_NAME]] = (
            pandas.DataFrame(rarest_word_and_frequency.to_list())
        )

        filtered_sentences_indexes = self._get_filtered_group_sentences_indices(sentences_df=sentences_df)
        sentences_df = sentences_df.loc[filtered_sentences_indexes].sort_values(
            [RAREST_WORD_FREQ_COL_NAME, RAREST_WORD_COL_NAME], ascending=False
        )

        _logger.info(
            f"Done, {len(sentences_df)} out of {len(self.original_sentences_df)} entries kept."
        )
        return sentences_df

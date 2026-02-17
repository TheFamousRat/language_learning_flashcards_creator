from __future__ import annotations
import json
from typing import cast
from deck_generation.bin.config import DeckGeneratorConfig
from deck_generation.constants import (
    AUDIO_FILE_COL_NAME,
    ORIGINAL_ID_COL_NAME,
    ORIGINAL_SENTENCE_COL_NAME,
    TRANSLATED_ID_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
)
from deck_generation.data_generation.kokoro_sentence_audio_generator import (
    KokoroSentenceAudioGenerator,
)
from deck_generation.data_generation.sentence_filterer import SentenceFilterer


import genanki  # type: ignore[import-untyped]
import numpy as np
import pandas


from pathlib import Path

import logging

logging.basicConfig()

_logger = logging.getLogger(name=__file__)
_logger.setLevel(level=logging.DEBUG)


class AnkiDeckGenerator:
    def __init__(
        self,
        deck_name: str,
        deck_output_folder: Path,
        sentences_filterer: SentenceFilterer,
        config: DeckGeneratorConfig,
    ) -> None:
        self.deck_name = deck_name
        self.deck_output_folder = deck_output_folder
        self.deck_sentences_data_file_path = self.deck_output_folder / "sentences.csv"
        self.audio_files_folder_path = self.deck_output_folder / "audio"
        self.config_file = self.deck_output_folder / "used_config.json"

        self.sentences_filterer = sentences_filterer

        self.config = config
        self.note_models = [
            note_model
            for note_model, _proportion in self.config.note_types_and_target_proportion
        ]
        self.proportions = [
            proportion
            for _note_model, proportion in self.config.note_types_and_target_proportion
        ]

        self.audio_generator = KokoroSentenceAudioGenerator(
            config=self.config.audio_generation_config
        )

    @classmethod
    def from_tatoeba_file(
        cls,
        deck_name: str,
        deck_output_folder: Path,
        tatoeba_sentences_file_path: Path,
        word_frequency_file_path: Path,
        config: DeckGeneratorConfig,
    ) -> AnkiDeckGenerator:
        filterer = SentenceFilterer.from_tatoeba_file(
            sentences_filepath=tatoeba_sentences_file_path,
            word_frequency_file_path=word_frequency_file_path,
        )

        return AnkiDeckGenerator(
            deck_name=deck_name,
            deck_output_folder=deck_output_folder,
            sentences_filterer=filterer,
            config=config,
        )

    def _generate_audio_files(
        self,
        sentences_df: pandas.DataFrame,
        previous_config: DeckGeneratorConfig | None,
    ) -> None:
        if not self.audio_files_folder_path.exists():
            self.audio_files_folder_path.mkdir(parents=True)

        configs_mismatch: bool = (
            True
            if previous_config is None
            else (
                previous_config.audio_generation_config
                != self.config.audio_generation_config
            )
        )

        sentences_df[AUDIO_FILE_COL_NAME] = sentences_df[ORIGINAL_ID_COL_NAME].apply(
            lambda sentence_id: str(self.audio_files_folder_path / f"{sentence_id}.mp3")
        )

        self.audio_generator.generate_sentences_audio(
            sentences=cast(
                list[str], sentences_df[ORIGINAL_SENTENCE_COL_NAME].tolist()
            ),
            sentences_ids=cast(list[int], sentences_df[ORIGINAL_ID_COL_NAME].tolist()),
            sentences_paths=cast(list[str], sentences_df[AUDIO_FILE_COL_NAME].tolist()),
            overwrite_existing_files=configs_mismatch,
        )

    def generate_deck_data(self) -> None:
        previous_config: DeckGeneratorConfig | None = None
        if self.config_file.exists():
            with open(file=self.config_file, mode="r") as f:
                previous_config = DeckGeneratorConfig.from_json(json.load(fp=f))

        sentences_df = self.sentences_filterer.get_filtered_sentences_df(
            config=self.config.sentence_filtering_config
        )
        sentences_df = sentences_df.reset_index(drop=True)

        self._generate_audio_files(
            sentences_df=sentences_df, previous_config=previous_config
        )

        sentences_df.to_csv(
            path_or_buf=self.deck_sentences_data_file_path,
            columns=[
                ORIGINAL_ID_COL_NAME,
                ORIGINAL_SENTENCE_COL_NAME,
                TRANSLATED_ID_COL_NAME,
                TRANSLATED_SENTENCE_COL_NAME,
                "rarest_word",
                "rarest_word_freq",
                AUDIO_FILE_COL_NAME,
            ],
            index=False,
        )

        with open(file=self.deck_output_folder / "used_config.json", mode="w") as f:
            json.dump(obj=self.config.to_json(), fp=f, indent=4)

    def make_deck(self) -> None:
        # TODO: Add condition to check whether the data needs to be regenerated or not
        self.generate_deck_data()

        deck_data_df = pandas.read_csv(
            filepath_or_buffer=self.deck_sentences_data_file_path
        )

        note_model_indices = self._get_note_model_indices(
            proportions=self.proportions, notes_count=len(deck_data_df)
        )

        my_deck = genanki.Deck(deck_id=2010120120, name=self.deck_name)

        notes_data = deck_data_df.apply(
            lambda row: self.note_models[
                note_model_indices[row.name]
            ].make_note_from_data(row=row),
            axis=1,
        )
        for note in notes_data:
            my_deck.add_note(note=note)

        package = genanki.Package(deck_or_decks=my_deck)
        package.media_files = deck_data_df[AUDIO_FILE_COL_NAME].values

        _logger.info("Saving deck...")
        package.write_to_file(file=self.deck_output_folder / f"{self.deck_name}.apkg")

        _logger.info("Done.")

    def _get_note_model_indices(
        self, proportions: list[float], notes_count: int
    ) -> list[int]:
        # TODO: Répartition des types de cartes déterministe
        proportions_np = np.array(proportions)
        proportions_np = proportions_np / proportions_np.sum()

        indices = np.arange(notes_count, dtype=np.int32)
        np.random.shuffle(indices)

        splits_indices = np.split(
            indices,
            np.round(np.cumsum(proportions_np)[:-1] * (len(indices) - 1)).astype(int),
        )
        note_model_indices = np.zeros_like(indices, dtype=np.int32)
        for note_model_idx, split_indices in enumerate(splits_indices):
            note_model_indices[split_indices] = note_model_idx

        return note_model_indices

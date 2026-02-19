from __future__ import annotations
import json
from typing import cast
import plotly.express as px  # type: ignore[import-untyped]
import genanki  # type: ignore[import-untyped]
import numpy as np
import pandas
from pathlib import Path


from deck_generation.bin.config import DeckGeneratorConfig
from deck_generation.constants import (
    AUDIO_FILE_COL_NAME,
    TARGET_ID_COL_NAME,
    TARGET_SENTENCE_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
    SENTENCES_WORDS_COL_NAME,
    RAREST_WORD_COL_NAME,
    RAREST_WORD_FREQ_COL_NAME,
)
from deck_generation.data_generation.kokoro_sentence_audio_generator import (
    KokoroSentenceAudioGenerator,
)
from deck_generation.data_generation.note_models import NoteModel
from deck_generation.data_generation.sentence_filterer import SentenceFilterer

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
        translated_language_code: str,
        target_language_code: str,
    ) -> None:
        self.deck_name = deck_name
        self.deck_output_folder = deck_output_folder
        self.deck_sentences_data_file_path = self.deck_output_folder / "sentences.json"
        self.audio_files_folder_path = self.deck_output_folder / "audio"
        self.config_file = self.deck_output_folder / "used_config.json"

        self.sentences_filterer = sentences_filterer

        self.config = config

        self.audio_generator = KokoroSentenceAudioGenerator(
            config=self.config.audio_generation_config
        )

        self.translated_language_code = translated_language_code
        self.target_language_code = target_language_code

    @classmethod
    def from_tatoeba_file(
        cls,
        deck_name: str,
        deck_output_folder: Path,
        tatoeba_sentences_file_path: Path,
        word_frequency_file_path: Path,
        config: DeckGeneratorConfig,
        translated_language_code: str,
        target_language_code: str,
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
            translated_language_code=translated_language_code,
            target_language_code=target_language_code,
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

        sentences_df[AUDIO_FILE_COL_NAME] = sentences_df[TARGET_ID_COL_NAME].apply(
            lambda sentence_id: str(self.audio_files_folder_path / f"{sentence_id}.mp3")
        )

        self.audio_generator.generate_sentences_audio(
            sentences=cast(list[str], sentences_df[TARGET_SENTENCE_COL_NAME].tolist()),
            sentences_ids=cast(list[int], sentences_df[TARGET_ID_COL_NAME].tolist()),
            sentences_paths=cast(list[str], sentences_df[AUDIO_FILE_COL_NAME].tolist()),
            overwrite_existing_files=configs_mismatch,
        )

    def generate_deck_data(self) -> None:
        previous_config: DeckGeneratorConfig | None = None
        if self.config_file.exists():
            with open(file=self.config_file, mode="r") as f:
                previous_config = DeckGeneratorConfig.from_json(json.load(fp=f))

        # TODO: Add condition to check whether the data needs to be regenerated or not
        sentences_df = self.sentences_filterer.get_filtered_sentences_df(
            config=self.config.sentence_filtering_config
        )
        sentences_df = sentences_df.reset_index(drop=True)

        self._generate_audio_files(
            sentences_df=sentences_df, previous_config=previous_config
        )

        sentences_df[
            [
                TARGET_ID_COL_NAME,
                TARGET_SENTENCE_COL_NAME,
                TRANSLATED_SENTENCE_COL_NAME,
                SENTENCES_WORDS_COL_NAME,
                RAREST_WORD_COL_NAME,
                RAREST_WORD_FREQ_COL_NAME,
                AUDIO_FILE_COL_NAME,
            ]
        ].to_json(
            path_or_buf=self.deck_sentences_data_file_path,
            force_ascii=False,
            indent=1,
            orient="records",
        )

        with open(file=self.deck_output_folder / "used_config.json", mode="w") as f:
            json.dump(obj=self.config.to_json(), fp=f, indent=4)

    def make_deck(self) -> None:
        self.generate_deck_data()

        deck_data_df = pandas.read_json(
            path_or_buf=self.deck_sentences_data_file_path,
        )

        cards_note_model = self._get_sentences_note_models(
            deck_data_df=deck_data_df,
        )

        # TODO: Set ID as hash from deck name maybe ?
        my_deck = genanki.Deck(deck_id=2010120120, name=self.deck_name)

        for note_model, (_row_index, row) in zip(
            cards_note_model, deck_data_df.iterrows()
        ):
            note = note_model.make_note_from_data(
                row=row,
                target_language_code=self.target_language_code,
                translated_language_code=self.translated_language_code,
            )
            my_deck.add_note(note=note)

        package = genanki.Package(deck_or_decks=my_deck)
        package.media_files = deck_data_df[AUDIO_FILE_COL_NAME].values

        _logger.info("Saving deck...")
        package.write_to_file(file=self.deck_output_folder / f"{self.deck_name}.apkg")

        _logger.info("Done.")

    @classmethod
    def _display_running_model_proportions(
        cls, models: list[NoteModel], note_model_indices: np.ndarray
    ) -> None:
        running_models_counts = np.zeros(
            shape=(len(models), len(note_model_indices)), dtype=np.int32
        )
        for model_idx, _model in enumerate(models):
            relevant_indices = np.where(note_model_indices == model_idx)[0]
            running_models_counts[model_idx, relevant_indices] = 1

        running_counts_df = pandas.DataFrame(
            running_models_counts.T,
            columns=[model.__class__.__name__ for model in models],
        )

        px.histogram(
            running_counts_df,
            x=running_counts_df.index,
            y=[c for c in running_counts_df.columns],
            cumulative=True,
            barnorm="percent",
            nbins=len(note_model_indices) // 20,
            title="Running proportion of cards types with deck progression",
        ).show()

    def _get_sentences_note_models(
        self,
        deck_data_df: pandas.DataFrame,
    ) -> list[NoteModel]:
        models: list[NoteModel] = [
            self.config.translating_note_model,
            self.config.reading_note_model,
            self.config.listening_note_model,
        ]
        models_target_proportions = np.array(
            [
                self.config.translating_notes_proportion,
                self.config.reading_notes_proportion,
                self.config.listening_notes_proportion,
            ]
        )
        valid_model_notes_masks: list[list[bool]] = [
            model.get_valid_sentence_masks(sentences_df=deck_data_df)
            for model in models
        ]

        models_notes_count = np.zeros_like(models_target_proportions, dtype=np.int32)
        note_model_indices = np.full(len(deck_data_df), fill_value=-1, dtype=np.int32)
        for note_index in range(len(deck_data_df)):
            potential_running_proportions = (models_notes_count + 1) / (note_index + 1)
            proportion_distance = (
                potential_running_proportions / models_target_proportions
            ) - 1

            for closest_model_index in proportion_distance.argsort():
                if valid_model_notes_masks[closest_model_index][note_index]:
                    note_model_indices[note_index] = closest_model_index
                    models_notes_count[closest_model_index] += 1
                    break

            if note_model_indices[note_index] == -1:
                raise RuntimeError(
                    f"Could not find a card type for sentence #{note_index} (ID: {deck_data_df[TARGET_ID_COL_NAME][note_index]})"
                )

        if self.config.plot_running_card_types_proportions:
            self._display_running_model_proportions(
                models=models, note_model_indices=note_model_indices
            )

        note_models: list[NoteModel] = [models[0]] * len(deck_data_df)
        for model_idx, model in enumerate(models[1:], start=1):
            model_note_indices = np.where(note_model_indices == model_idx)[0]
            for note_model_idx in model_note_indices:
                note_models[note_model_idx] = model

        return note_models

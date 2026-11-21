from __future__ import annotations

import logging
from pathlib import Path

import pandas

from deck_generation.constants import (
    GENERATED_DECK_DATA_DIR,
    SOURCE_DATA_DIR,
)
from deck_generation.deck_generation.anki_deck_generator import AnkiDeckGenerator
from deck_generation.deck_generation.config import (
    AudioGenerationConfig,
    DeckGeneratorConfig,
    SentenceFilteringConfig,
)
from deck_generation.deck_generation.note_models import (
    ListeningNoteModel,
    ReadingNoteModel,
    TranslatingNoteModel,
)
from deck_generation.preprocessing.sentences_dataset import (
    DatasetIndex,
    DatasetLoadingConfig,
    SentencesDataset,
)

logging.basicConfig()

_logger = logging.getLogger(name=__name__)
_logger.setLevel(level=logging.DEBUG)


def main() -> None:
    tatoeba_dataset = SentencesDataset.from_loading_config(
        loading_config=DatasetLoadingConfig(
            source_sentences_file_path=SOURCE_DATA_DIR
            / "it_fr/Sentence pairs in Italian-French - 2026-02-07.tsv",
            separator="	",
            has_header=False,
            original_sentences_column=1,
            translated_sentences_column=3,
        )
    )

    def _make_full_audio_files_path_from_relative_column(
        dataset_index: DatasetIndex, sentences_dataset: SentencesDataset
    ) -> pandas.Series:
        audio_files = sentences_dataset.load_new_dataset_dataframe_columns(
            dataset_index=dataset_index, columns_to_load=["path"]
        )
        return audio_files.apply(
            lambda file_path: sentences_dataset.loading_config_by_dataset_index[
                dataset_index
            ].source_sentences_file_path
            / "clips"
            / file_path
        )

    mozilla_dataset = SentencesDataset.from_loading_config(
        loading_config=DatasetLoadingConfig(
            source_sentences_file_path=SOURCE_DATA_DIR
            / "it_fr/cv-corpus-27.0-2026-09-11/it/validated.tsv",
            separator="	",
            has_header=True,
            original_sentences_column="sentence",
            audio_files_generator=_make_full_audio_files_path_from_relative_column,
        )
    )

    full_dataset = mozilla_dataset + tatoeba_dataset

    deck_generator = AnkiDeckGenerator.from_dataset(
        deck_name="Deck français-italien",
        deck_output_folder=GENERATED_DECK_DATA_DIR,
        sentences_dataset=full_dataset,
        word_frequency_file_path=Path("data/it_fr/word_freq_it.csv"),
        target_language_code="it",
        translated_language_code="fr",
        config=DeckGeneratorConfig(
            sentence_filtering_config=SentenceFilteringConfig(
                min_word_count=4,
                max_word_count=20,
                max_sentences_count_per_new_word=2,
                max_sentence_similarity_per_new_word=0.25,
            ),
            audio_generation_config=AudioGenerationConfig(
                language_code="i",
                voices=[
                    "im_nicola",
                    "if_sara",
                ],
            ),
            reading_note_model=ReadingNoteModel(),
            reading_notes_proportion=0.4,
            listening_note_model=ListeningNoteModel(),
            listening_notes_proportion=0.4,
            translating_note_model=TranslatingNoteModel(),
            translating_notes_proportion=0.2,
            plot_running_card_types_proportions=False,
        ),
    )

    deck_generator.make_deck()


if __name__ == "__main__":
    main()

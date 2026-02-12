from __future__ import annotations
from pathlib import Path

from deck_generation.data_generation.anki_deck_generator import AnkiDeckGenerator
from deck_generation.bin.config import (
    AudioGenerationConfig,
    DeckGeneratorConfig,
    SentenceFilteringConfig,
)
from deck_generation.constants import (
    GENERATED_DECK_DATA_DIR,
)
from deck_generation.data_generation.note_models import (
    ListeningNoteModel,
    ReadingNoteModel,
    TranslatingNoteModel,
)
import logging

logging.basicConfig()

_logger = logging.getLogger(name=__file__)
_logger.setLevel(level=logging.DEBUG)


def main() -> None:
    # TODO: Ajouter un lien vers les mots du wikitionnaire
    sentences_filepath = Path("data/Sentence pairs in Italian-French - 2026-02-07.tsv")

    deck_generator = AnkiDeckGenerator.from_tatoeba_file(
        deck_name="Deck français-italien",
        deck_output_folder=GENERATED_DECK_DATA_DIR / sentences_filepath.stem,
        tatoeba_sentences_file_path=sentences_filepath,
        word_frequency_file_path=Path("data/word_freq_it.csv"),
        config=DeckGeneratorConfig(
            sentence_filtering_config=SentenceFilteringConfig(
                min_word_count=4,
                max_word_count=20,
                max_sentences_count_per_new_word=2,
                only_proper_nouns_capitalized=True,
            ),
            audio_generation_config=AudioGenerationConfig(
                language_code="i",
                voices=[
                    "im_nicola",
                    "if_sara",
                ],
            ),
            note_types_and_target_proportion=[
                (ReadingNoteModel(), 0.4),
                (ListeningNoteModel(), 0.4),
                (TranslatingNoteModel(), 0.2),
            ],
        ),
    )

    deck_generator.make_deck()


if __name__ == "__main__":
    main()

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

from deck_generation.data_generation.note_models import NoteModel


@dataclass(frozen=True)
class SentenceFilteringConfig(DataClassJsonMixin):
    min_word_count: int
    max_word_count: int
    max_sentences_count_per_new_word: int
    only_proper_nouns_capitalized: bool


@dataclass(frozen=True)
class AudioGenerationConfig(DataClassJsonMixin):
    voices: list[str]
    language_code: str


@dataclass(frozen=True)
class DeckGeneratorConfig(DataClassJsonMixin):
    sentence_filtering_config: SentenceFilteringConfig
    audio_generation_config: AudioGenerationConfig
    note_types_and_target_proportion: list[tuple[NoteModel, float]]

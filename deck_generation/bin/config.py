from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json

from deck_generation.data_generation.note_models import (
    ListeningNoteModel,
    ReadingNoteModel,
    TranslatingNoteModel,
)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class SentenceFilteringConfig(DataClassJsonMixin):
    min_word_count: int
    max_word_count: int
    max_sentences_count_per_new_word: int
    only_proper_nouns_capitalized: bool


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class AudioGenerationConfig(DataClassJsonMixin):
    voices: list[str]
    language_code: str


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class DeckGeneratorConfig:
    sentence_filtering_config: SentenceFilteringConfig
    audio_generation_config: AudioGenerationConfig

    reading_note_model: ReadingNoteModel = field(default_factory=ReadingNoteModel)
    reading_notes_proportion: float = 0.4
    listening_note_model: ListeningNoteModel = field(default_factory=ListeningNoteModel)
    listening_notes_proportion: float = 0.4
    translating_note_model: TranslatingNoteModel = field(
        default_factory=TranslatingNoteModel
    )
    translating_notes_proportion: float = 0.2

    plot_running_card_types_proportions: bool = False

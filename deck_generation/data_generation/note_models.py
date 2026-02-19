from dataclasses import dataclass
from pathlib import Path
from dataclasses_json import DataClassJsonMixin
import genanki  # type: ignore[import-untyped]
import pandas

from deck_generation.constants import (
    AUDIO_FILE_COL_NAME,
    TARGET_SENTENCE_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
    SENTENCES_WORDS_COL_NAME,
    RAREST_WORD_COL_NAME,
)


@dataclass(frozen=True)
class NoteModelFields(DataClassJsonMixin):
    model_id: int
    name: str
    fields: list[dict]
    templates: list[dict]
    css: str

    @property
    def model(self) -> genanki.Model:
        return genanki.Model(
            model_id=self.model_id,
            name=self.name,
            fields=self.fields,
            templates=self.templates,
            css=self.css,
        )


@dataclass()
class NoteModel(DataClassJsonMixin):
    model_fields: NoteModelFields | None = None

    def make_note_from_data(
        self,
        row: pandas.Series,
        target_language_code: str,
        translated_language_code: str,
    ) -> genanki.Note:
        assert self.model_fields is not None
        wiki_hyperlinks: list[str] = [
            f'<a href="https://{translated_language_code}.wiktionary.org/wiki/{word if word_idx > 0 else word.lower()}#{target_language_code}">{word}</a>'
            for sentence in row[SENTENCES_WORDS_COL_NAME]
            for word_idx, word in enumerate(sentence)
        ]

        return genanki.Note(
            model=self.model_fields.model,
            fields=[
                row[TARGET_SENTENCE_COL_NAME],
                row[TRANSLATED_SENTENCE_COL_NAME],
                f"[sound:{Path(row[AUDIO_FILE_COL_NAME]).name}]",
                " ".join(wiki_hyperlinks),
            ],
        )

    def get_valid_sentence_masks(self, sentences_df: pandas.DataFrame) -> list[bool]:
        return len(sentences_df) * [True]


@dataclass()
class ReadingNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392319,
            name="Reading Note",
            fields=[
                {"name": "Target sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
                {"name": "Links"},
            ],
            templates=[
                {
                    "name": "Reading",
                    "qfmt": '<div align="center">{{Target sentence}}<br>{{MyMedia}}</div>',
                    "afmt": """
{{FrontSide}}
<hr id="answer">
<div align="center">{{Native sentence}}</div>

<br>Wikitionaire: {{Links}}
""",
                },
            ],
            css="",
        )

    def get_valid_sentence_masks(self, sentences_df: pandas.DataFrame) -> list[bool]:
        return len(sentences_df) * [True]


@dataclass()
class ListeningNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392320,
            name="Listening Note",
            fields=[
                {"name": "Target sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
                {"name": "Links"},
            ],
            templates=[
                {
                    "name": "Listening",
                    "qfmt": """<br>
<div align="center">{{MyMedia}}</div>""",
                    "afmt": """
{{FrontSide}}<hr id="answer">
<div align="center">{{Target sentence}}<br>{{Native sentence}}</div>

<br>Wikitionaire: {{Links}}
""",
                },
            ],
            css="",
        )

    def get_valid_sentence_masks(self, sentences_df: pandas.DataFrame) -> list[bool]:
        return len(sentences_df) * [True]


@dataclass()
class TranslatingNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392321,
            name="Translating Note",
            fields=[
                {"name": "Target sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
                {"name": "Links"},
            ],
            templates=[
                {
                    "name": "Translating",
                    "qfmt": """<div align="center">{{Native sentence}}</div>""",
                    "afmt": """
{{FrontSide}}<hr id="answer"><div align="center">{{MyMedia}}<br>{{Target sentence}}</div>

<br>Wikitionaire: {{Links}}
""",
                },
            ],
            css="",
        )

    def get_valid_sentence_masks(self, sentences_df: pandas.DataFrame) -> list[bool]:
        word_introduction_count = (
            sentences_df.groupby(RAREST_WORD_COL_NAME, sort=False)[RAREST_WORD_COL_NAME]
            .rolling(window=2, min_periods=1)
            .count()
            .astype(int)
            .reset_index(drop=True)
        )
        sentence_rarest_word_already_introduced_mask = (
            word_introduction_count > 1
        ).tolist()

        return sentence_rarest_word_already_introduced_mask

from dataclasses import dataclass
from pathlib import Path
from dataclasses_json import DataClassJsonMixin
import genanki  # type: ignore[import-untyped]
import pandas

from deck_generation.constants import (
    AUDIO_FILE_COL_NAME,
    ORIGINAL_SENTENCE_COL_NAME,
    TRANSLATED_SENTENCE_COL_NAME,
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

    def make_note_from_data(self, row: pandas.Series):
        assert self.model_fields is not None
        return genanki.Note(
            model=self.model_fields.model,
            fields=[
                row[ORIGINAL_SENTENCE_COL_NAME],
                row[TRANSLATED_SENTENCE_COL_NAME],
                f"[sound:{Path(row[AUDIO_FILE_COL_NAME]).name}]",
            ],
        )


@dataclass()
class ReadingNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392319,
            name="Reading Note",
            fields=[
                {"name": "Original sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
            ],
            templates=[
                {
                    "name": "Reading",
                    "qfmt": "{{Original sentence}}<br>{{MyMedia}}",
                    "afmt": '{{FrontSide}}<hr id="answer">{{Native sentence}}',
                },
            ],
            css="",
        )


@dataclass()
class ListeningNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392320,
            name="Listening Note",
            fields=[
                {"name": "Original sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
            ],
            templates=[
                {
                    "name": "Listening",
                    "qfmt": "<br>{{MyMedia}}",
                    "afmt": '{{FrontSide}}<hr id="answer">{{Original sentence}}<br>{{Native sentence}}',
                },
            ],
            css="",
        )


@dataclass()
class TranslatingNoteModel(NoteModel):
    def __post_init__(self) -> None:
        self.model_fields = NoteModelFields(
            model_id=1607392321,
            name="Translating Note",
            fields=[
                {"name": "Original sentence"},
                {"name": "Native sentence"},
                {"name": "MyMedia"},
            ],
            templates=[
                {
                    "name": "Translating",
                    "qfmt": "{{Native sentence}}",
                    "afmt": '{{FrontSide}}<br>{{MyMedia}}<hr id="answer">{{Original sentence}}',
                },
            ],
            css="",
        )

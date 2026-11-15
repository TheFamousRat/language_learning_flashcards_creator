from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas


class SentencesDataset:
    ORIGINAL_SENTENCES_COLUMN = "original_sentences"
    TRANSLATED_SENTENCES_COLUMN = "translated_sentences"
    SENTENCES_AUDIO_COLUMN = "original_sentences_audio"

    def __init__(
        self,
        source_sentences_file_path: Path,
        sentences_df: pandas.DataFrame,
        separator: str,
        has_header: bool,
        original_sentences_column: str,
        translated_sentences_column: str | None = None,
        audio_files_column: str | None = None,
        audio_files_generator: (
            Callable[[pandas.DataFrame], pandas.Series] | None
        ) = None,
    ) -> None:
        self.source_sentences_file_path = source_sentences_file_path
        self.sentences_df = sentences_df
        self.separator = separator
        self.has_header = has_header
        self.original_sentences_column = original_sentences_column
        self.translated_sentences_column = translated_sentences_column
        self.audio_files_column = audio_files_column
        self.audio_files_generator = audio_files_generator

    @property
    def original_sentences(self) -> pandas.Series[str]:
        return self.sentences_df[self.ORIGINAL_SENTENCES_COLUMN]

    @property
    def translated_sentences(self) -> pandas.Series[str] | None:
        if self.TRANSLATED_SENTENCES_COLUMN in self.sentences_df.columns:
            return self.sentences_df[self.TRANSLATED_SENTENCES_COLUMN]

        if self.translated_sentences_column is not None:
            self.sentences_df[self.TRANSLATED_SENTENCES_COLUMN] = (
                self._load_new_dataframe_columns(
                    columns_to_load=[self.translated_sentences_column]
                )
            )

        return self.sentences_df.get(
            key=[self.TRANSLATED_SENTENCES_COLUMN], default=None
        )

    @property
    def sentences_audio(self) -> pandas.Series[str] | None:
        if self.SENTENCES_AUDIO_COLUMN in self.sentences_df.columns:
            return self.sentences_df[self.SENTENCES_AUDIO_COLUMN]

        if self.audio_files_generator is not None:
            self.sentences_df[self.SENTENCES_AUDIO_COLUMN] = self.audio_files_generator(
                self.sentences_df
            )
        elif self.audio_files_column is not None:
            self.sentences_df[self.SENTENCES_AUDIO_COLUMN] = (
                self._load_new_dataframe_columns(
                    columns_to_load=[self.audio_files_column]
                )
            )

        return self.sentences_df.get(key=[self.SENTENCES_AUDIO_COLUMN], default=None)

    def _load_new_dataframe_columns(
        self, columns_to_load: list[str]
    ) -> pandas.DataFrame:
        return self._load_columns_from_dataframe_file(
            dataframe_file=self.source_sentences_file_path,
            separator=self.separator,
            has_header=self.has_header,
            columns_to_load=columns_to_load,
            index=self.sentences_df.index,
        )

    @classmethod
    def _load_columns_from_dataframe_file(
        cls,
        dataframe_file: Path,
        separator: str,
        has_header: bool,
        columns_to_load: list[str],
        index: pandas.Index | None = None,
    ) -> pandas.DataFrame:
        header: Literal["infer"] | None = "infer" if has_header else None
        if index is None:
            return pandas.read_csv(
                filepath_or_buffer=dataframe_file,
                usecols=columns_to_load,
                sep=separator,
                header=header,
            )

        parts: list[pandas.DataFrame] = []
        for chunk in pandas.read_csv(
            filepath_or_buffer=dataframe_file,
            usecols=columns_to_load,
            sep=separator,
            header=header,
            chunksize=200_000,
        ):
            filtered_chunk = chunk.loc[chunk.index.isin(index)]
            parts.append(filtered_chunk)

        new_columns_df = pandas.concat(parts)
        new_columns_df = new_columns_df.reindex(index)

        return new_columns_df

    @classmethod
    def from_sentences_file(
        cls,
        sentences_file_path: Path,
        separator: str,
        has_header: bool,
        original_sentences_column: str,
        translated_sentences_column: str | None = None,
        audio_files_column: str | None = None,
        audio_files_generator: (
            Callable[[pandas.DataFrame], pandas.Series] | None
        ) = None,
    ) -> SentencesDataset:
        sentences_df = cls._load_columns_from_dataframe_file(
            dataframe_file=sentences_file_path,
            separator=separator,
            has_header=has_header,
            columns_to_load=[original_sentences_column],
        )
        sentences_df[cls.ORIGINAL_SENTENCES_COLUMN] = sentences_df[
            original_sentences_column
        ]

        return SentencesDataset(
            source_sentences_file_path=sentences_file_path,
            separator=separator,
            has_header=has_header,
            sentences_df=sentences_df,
            original_sentences_column=original_sentences_column,
            translated_sentences_column=translated_sentences_column,
            audio_files_column=audio_files_column,
            audio_files_generator=audio_files_generator,
        )

    def append_dataset(self, other: SentencesDataset) -> SentencesDataset:
        pass


def main() -> None:
    sentences_file = Path(
        "/home/thefamousrat/Documents/Projets/language_learning_flashcards_creator/data/it_fr/cv-corpus-27.0-2026-09-11/it/validated.tsv"
    )

    def get_audio_files_fullpath(
        sentences_df: pandas.DataFrame, sentences_file_path: Path
    ) -> pandas.Series:
        audio_files = pandas.read_csv(
            filepath_or_buffer=sentences_file_path, sep="	", usecols=["path"]
        )["path"]
        audio_files = audio_files.iloc[sentences_df.index]
        audio_files_abs = audio_files.apply(
            lambda file_name: sentences_file_path.parent / "clips" / file_name
        )

        return audio_files_abs

    dataset_mozilla = SentencesDataset.from_sentences_file(
        sentences_file_path=sentences_file,
        separator="	",
        original_sentences_column="sentence",
        audio_files_generator=lambda sentences_df: get_audio_files_fullpath(
            sentences_df=sentences_df, sentences_file_path=sentences_file
        ),
    )

    dataset_tatoeba = SentencesDataset.from_sentences_file(
        sentences_file_path=Path(
            "/home/thefamousrat/Documents/Projets/language_learning_flashcards_creator/data/it_fr/Sentence pairs in Italian-French - 2026-02-07.tsv"
        ),
        separator="	",
        original_sentences_column=1,
        translated_sentences_column=3,
    )

    dataset_tatoeba.append_dataset(other=dataset_tatoeba)
    dataset_tatoeba.translated_sentences



if __name__ == "__main__":
    main()

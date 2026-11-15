from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Literal

import pandas

ColumnGenerator = Callable[["SentencesDataset"], pandas.Series]


class SentencesDataset:
    ORIGINAL_SENTENCES_COLUMN = "original_sentences"
    TRANSLATED_SENTENCES_COLUMN = "translated_sentences"
    SENTENCES_AUDIO_COLUMN = "original_sentences_audio"
    _DATASET_INDEX_COLUMN = "dataset_index"
    _ORIGINAL_FILE_ROWS_COLUMN = "original_file_row"

    def __init__(
        self,
        source_sentences_file_path: Path,
        sentences_df: pandas.DataFrame,
        separator: str,
        has_header: bool,
        original_sentences_column: Hashable,
        translated_sentences_column: Hashable | None = None,
        audio_files_column: Hashable | None = None,
        audio_files_generator: ColumnGenerator | None = None,
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
        return self._lazy_load_column_if_it_does_not_exist(
            column_name=self.TRANSLATED_SENTENCES_COLUMN,
            column_in_file=self.translated_sentences_column,
        )

    @property
    def sentences_audio(self) -> pandas.Series[str] | None:
        return self._lazy_load_column_if_it_does_not_exist(
            column_name=self.SENTENCES_AUDIO_COLUMN,
            column_in_file=self.audio_files_column,
            column_generator=self.audio_files_generator,
        )

    def _lazy_load_column_if_it_does_not_exist(
        self,
        column_name: Hashable,
        column_in_file: Hashable | None = None,
        column_generator: ColumnGenerator | None = None,
    ) -> pandas.Series | None:
        if column_name not in self.sentences_df:
            if column_in_file is not None:
                loaded_column = self.load_new_dataframe_columns(
                    columns_to_load=[column_in_file]
                )
                self.sentences_df[column_name] = loaded_column
            elif column_generator is not None:
                generated_column = column_generator(self)
                self.sentences_df[column_name] = generated_column

        column_content = (
            None
            if column_name not in self.sentences_df.columns
            else self.sentences_df[column_name]
        )
        return column_content

    def load_new_dataframe_columns(
        self, columns_to_load: list[Hashable]
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
        columns_to_load: list[Hashable],
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
        original_sentences_column: Hashable,
        translated_sentences_column: Hashable | None = None,
        audio_files_column: Hashable | None = None,
        audio_files_generator: ColumnGenerator | None = None,
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
        sentences_df[cls._DATASET_INDEX_COLUMN] = 1

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

    def append_dataset(self, other: SentencesDataset) -> None:
        pass
